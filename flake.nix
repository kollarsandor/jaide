{
  description = "JAIDE V40: Root-Level, Non-Transformer LLM Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils, ... }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "jaide-v40";
          src = ./.;
          buildInputs = with pkgs; [ zig ];
          buildPhase = ''
            echo "JAIDE v40 Build System Ready"
            echo "Development environment configured"
          '';
          installPhase = ''
            mkdir -p $out/bin
            echo "#!/bin/sh" > $out/bin/jaide
            echo "echo 'JAIDE v40 - Root-Level LLM System'" >> $out/bin/jaide
            echo "echo 'Run: zig build to compile the system'" >> $out/bin/jaide
            chmod +x $out/bin/jaide
          '';
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [ 
            zig
            lean4
            isabelle
            agda
            tlaplus
            viper
            nodejs
            bash
          ];
          shellHook = ''
            echo "========================================="
            echo "JAIDE v40 Development Environment"
            echo "========================================="
            echo "Build commands:"
            echo "  zig build       - Build the system"
            echo "  zig build verify - Run all formal verifications"
            echo "========================================="
            echo "Verification tools available:"
            echo "  - Lean4 (RSF invertibility proofs)"
            echo "  - Isabelle/HOL (memory safety proofs)"
            echo "  - Agda (constructive proofs)"
            echo "  - Viper (memory safety verification)"
            echo "  - TLA+ (IPC liveness proofs)"
            echo "========================================="
          '';
        };
      }
    );
}
