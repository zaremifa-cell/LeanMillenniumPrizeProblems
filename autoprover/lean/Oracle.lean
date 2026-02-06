import Lean

open Lean
open Lean.Meta
open Lean.Elab
open Lean.Elab.Term

partial def exprSize : Expr → Nat
  | .bvar _ => 1
  | .fvar _ => 1
  | .mvar _ => 1
  | .sort _ => 1
  | .const _ _ => 1
  | .app f a => exprSize f + exprSize a + 1
  | .lam _ t b _ => exprSize t + exprSize b + 1
  | .forallE _ t b _ => exprSize t + exprSize b + 1
  | .letE _ t v b _ => exprSize t + exprSize v + exprSize b + 1
  | .lit _ => 1
  | .mdata _ e => exprSize e + 1
  | .proj _ _ e => exprSize e + 1

structure OracleAction where
  kind : String
  lemma : Option String := none
  direction : Option String := none

structure ContextEntry where
  name : String
  type : String

structure OracleRequest where
  goalExpr : String
  localContext : Array ContextEntry
  action : OracleAction

private def optStr (j : Json) (key : String) : Option String :=
  match Json.getObjVal? j key with
  | Except.ok v =>
      match Json.getStr? v with
      | Except.ok s => some s
      | Except.error _ => none
  | Except.error _ => none

private def parseAction (j : Json) : Except String OracleAction := do
  let kind ← (Json.getObjVal? j "kind" >>= Json.getStr?)
  let lemma := optStr j "lemma"
  let direction := optStr j "direction"
  return { kind := kind, lemma := lemma, direction := direction }

private def parseContextEntry (j : Json) : Except String ContextEntry := do
  let name ← (Json.getObjVal? j "name" >>= Json.getStr?)
  let ty ← (Json.getObjVal? j "type" >>= Json.getStr?)
  return { name := name, type := ty }

private def parseRequest (j : Json) : Except String OracleRequest := do
  let goal ← (Json.getObjVal? j "goal_expr" >>= Json.getStr?)
  let ctxJson ← Json.getObjVal? j "local_context"
  let ctxArr ← Json.getArr? ctxJson
  let ctx ← ctxArr.mapM parseContextEntry
  let actionJson ← Json.getObjVal? j "action"
  let action ← parseAction actionJson
  return { goalExpr := goal, localContext := ctx, action := action }

private def parseTerm (env : Environment) (s : String) : TermElabM Expr := do
  match Parser.runParserCategory env `term s with
  | Except.error e => throwError m!"parse error: {e}"
  | Except.ok stx => elabTerm stx none

partial def withLocalCtx (env : Environment) (ctx : List ContextEntry)
    (k : Array Expr → TermElabM α) : TermElabM α :=
  match ctx with
  | [] => k #[]
  | entry :: rest => do
      let ty ← parseTerm env entry.type
      let name := entry.name.toName
      Meta.withLocalDecl name BinderInfo.default ty fun fvar =>
        withLocalCtx env rest (fun fvars => k (fvars.push fvar))

private def goalListToJson (goals : List MVarId) : MetaM (Array String × Array Nat) := do
  let mut out : Array String := #[]
  let mut sizes : Array Nat := #[]
  for gid in goals do
    let ty ← gid.getType
    let fmt ← Meta.ppExpr ty
    out := out.push fmt.pretty
    sizes := sizes.push (exprSize ty)
  return (out, sizes)

private def localContextToJson (lctx : LocalContext) : MetaM (Array Json) := do
  lctx.foldlM (init := #[]) fun arr decl => do
    if decl.isImplementationDetail then
      return arr
    let fmt ← Meta.ppExpr decl.type
    let obj := Json.mkObj [
      ("name", Json.str decl.userName.toString),
      ("type", Json.str fmt.pretty)
    ]
    return arr.push obj

private def runAction (action : OracleAction) (mvarId : MVarId) : MetaM (List MVarId) := do
  match action.kind with
  | "apply" =>
      let lemmaName := (action.lemma.getD "").toName
      let expr ← mkConstWithFreshMVarLevels lemmaName
      mvarId.apply expr
  | "exact" =>
      let lemmaName := (action.lemma.getD "").toName
      let expr ← mkConstWithFreshMVarLevels lemmaName
      mvarId.assign expr
      return []
  | "intro" =>
      let (_, mvarId') ← mvarId.intro1
      return [mvarId']
  | "rw" =>
      let lemmaName := (action.lemma.getD "").toName
      let expr ← mkConstWithFreshMVarLevels lemmaName
      let target ← mvarId.getType
      let symm := action.direction == some "backward"
      let r ← mvarId.rewrite target expr (symm := symm)
      let mvarId' ← mvarId.replaceTargetEq r.eNew r.eqProof
      return mvarId' :: r.mvarIds
  | _ =>
      throwError m!"unknown action kind {action.kind}"

private def handleRequest (env : Environment) (req : OracleRequest) : IO Json := do
  let coreCtx : Core.Context := {
    fileName := "<oracle>"
    fileMap := default
  }
  let coreState : Core.State := { env := env }
  let action : TermElabM Json := do
    withLocalCtx env req.localContext.toList fun _fvars => do
      let goal ← parseTerm env req.goalExpr
      let mvar ← mkFreshExprMVar goal
      let mvarId := mvar.mvarId!
      let newGoals ← runAction req.action mvarId
      let (goalsJson, goalSizes) ← goalListToJson newGoals
      let ctxJson ←
        if let some first := newGoals.head? then
          let decl ← first.getDecl
          localContextToJson decl.lctx
        else
          pure #[]
      return Json.mkObj [
        ("ok", Json.bool true),
        ("goals", Json.arr (goalsJson.map Json.str)),
        ("goal_ast_sizes", Json.arr (goalSizes.map (fun n => Json.num (JsonNumber.fromNat n)))),
        ("context", Json.arr ctxJson)
      ]
  try
    let (result, _, _) ← (TermElabM.run' action).toIO coreCtx coreState
    return result
  catch e =>
    return Json.mkObj [
      ("ok", Json.bool false),
      ("error", Json.str (toString e))
    ]

partial def loop (env : Environment) : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let line ← stdin.getLine
  if line.isEmpty then
    return  -- EOF
  if line.trim.isEmpty then
    loop env
  else
    match Json.parse line with
    | Except.error e =>
        IO.println <| (Json.mkObj [("ok", Json.bool false), ("error", Json.str e)]).compress
        stdout.flush
        loop env
    | Except.ok j =>
        match parseRequest j with
        | Except.error e =>
            IO.println <| (Json.mkObj [("ok", Json.bool false), ("error", Json.str e)]).compress
            stdout.flush
            loop env
        | Except.ok req =>
            let resp ← handleRequest env req
            IO.println resp.compress
            stdout.flush
            loop env

def main : IO Unit := do
  let env ← importModules #[{ module := `Mathlib }] {} 0
  loop env
