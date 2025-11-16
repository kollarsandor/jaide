import Lake
open Lake DSL

package RSF_Verification where
  precompileModules := true

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.3.0"

@[default_target]
lean_lib RSF_Properties where
  srcDir := "."
