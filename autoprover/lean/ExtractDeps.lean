import Lean
import Mathlib

open Lean

structure Config where
  modulePrefix : String := "Mathlib.Data"
  kindFilter : Option String := none
  limit : Option Nat := none

def parseArgs : List String -> Config
  | [] => {}
  | "--module-prefix" :: v :: rest =>
      let cfg := parseArgs rest
      { cfg with modulePrefix := v }
  | "--kind" :: v :: rest =>
      let cfg := parseArgs rest
      { cfg with kindFilter := some v }
  | "--limit" :: v :: rest =>
      let cfg := parseArgs rest
      let n := v.toNat?
      { cfg with limit := n }
  | _ :: rest => parseArgs rest

partial def exprSize : Expr -> Nat
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

partial def collectConsts (e : Expr) (acc : NameSet := {}) : NameSet :=
  match e with
  | .const n _ => acc.insert n
  | .app f a => collectConsts a (collectConsts f acc)
  | .lam _ t b _ => collectConsts b (collectConsts t acc)
  | .forallE _ t b _ => collectConsts b (collectConsts t acc)
  | .letE _ t v b _ => collectConsts b (collectConsts v (collectConsts t acc))
  | .mdata _ e => collectConsts e acc
  | .proj _ _ e => collectConsts e acc
  | _ => acc

partial def tokensFromString (s : String) : Array String :=
  let mut acc : Array String := #[]
  let mut cur := ""
  for c in s.data do
    if c.isWhitespace then
      if cur != "" then
        acc := acc.push cur
        cur := ""
    else
      cur := cur.push c
  if cur != "" then
    acc := acc.push cur
  acc

private def constKind : ConstantInfo -> String
  | .thmInfo _ => "theorem"
  | .defnInfo _ => "definition"
  | .axiomInfo _ => "axiom"
  | .opaqueInfo _ => "opaque"
  | .inductiveInfo _ => "inductive"
  | .ctorInfo _ => "ctor"
  | .recInfo _ => "recursor"
  | .quotInfo _ => "quot"

private def constValue? : ConstantInfo -> Option Expr
  | .thmInfo v => some v.value
  | .defnInfo v => some v.value
  | .opaqueInfo v => some v.value
  | _ => none

private def moduleOf? (env : Environment) (decl : Name) : Option Name := do
  let midx ← env.getModuleIdxFor? decl
  env.header.moduleNames.get? midx

private def startsWithModule (moduleName : Name) (prefix : String) : Bool :=
  (toString moduleName).startsWith prefix

private def exprToString (env : Environment) (e : Expr) : IO String := do
  let ctxCore : Core.Context := { env := env }
  let sCore : Core.State := { env := env }
  let ((fmt, _), _) ← (Meta.ppExpr e).run |>.toIO ctxCore sCore
  pure fmt.pretty

private def jsonOfEntry (decl : Name) (moduleName : Name) (kind : String)
    (typeConsts valueConsts : NameSet) (typeTokens : Array String) (typeSize : Nat) (typeStr : String) : Json :=
  let arrOfNames (ns : NameSet) :=
    Json.arr <| (ns.toList.map (fun n => Json.str (toString n))).toArray
  Json.mkObj [
    ("decl_name", Json.str (toString decl)),
    ("module", Json.str (toString moduleName)),
    ("kind", Json.str kind),
    ("type_consts", arrOfNames typeConsts),
    ("value_consts", arrOfNames valueConsts),
    ("type_expr_ast_size", Json.num (JsonNumber.fromNat typeSize)),
    ("type_expr_tokens", Json.arr (typeTokens.map Json.str)),
    ("type_expr", Json.str typeStr)
  ]

def main (args : List String) : IO Unit := do
  let cfg := parseArgs args
  let env ← importModules #[{ module := `Mathlib }] {} 0
  let mut entries := env.constants.toList
  entries := entries.qsort (fun a b => (toString a.1) < (toString b.1))
  let mut count := 0
  for (decl, info) in entries do
    match moduleOf? env decl with
    | none => continue
    | some modName =>
      if !startsWithModule modName cfg.modulePrefix then
        continue
      let kind := constKind info
      match cfg.kindFilter with
      | some k => if k != kind then continue
      | none => pure ()
      let typeConsts := collectConsts info.type
      let valueConsts :=
        match constValue? info with
        | some v => collectConsts v
        | none => {}
      let typeStr ← exprToString env info.type
      let typeTokens := tokensFromString typeStr
      let typeSize := exprSize info.type
      let j := jsonOfEntry decl modName kind typeConsts valueConsts typeTokens typeSize typeStr
      IO.println j.compress
      count := count + 1
      match cfg.limit with
      | some lim => if count >= lim then break
      | none => pure ()
