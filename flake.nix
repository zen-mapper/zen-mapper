{
  description = "Mapper without the noise";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    python = pkgs.python3.withPackages (ps: [
      ps.numpy
      ps.pytest
      ps.scikit-learn
      ps.hypothesis
      ps.networkx
      ps.sphinx
      ps.myst-parser
    ]);
  in {
    formatter.${system} = pkgs.alejandra;

    packages.${system}.default = pkgs.python3Packages.callPackage ./nix/zen-mapper.nix {};

    overlays.default = final: prev: {
      pythonPackagesExtensions =
        (prev.pythonPackagesExtensions or [])
        ++ [
          (
            python-final: python-prev: {
              zen-mapper = python-final.callPackage ./nix/zen-mapper.nix {};
            }
          )
        ];
    };

    templates.default = {
      path = ./nix/templates/minimal;
      description = "Minimal example of using zen-mapper with flakes";
    };

    devShells.${system}.default = pkgs.mkShell {
      venvDir = ".venv";
      buildInputs = [
        pkgs.hatch
        pkgs.just
        pkgs.ruff
        python
      ];

      shellHook = ''
        if [ -z "$PYTHONPATH" ]
        then export PYTHONPATH=$(realpath ./src)
        else export PYTHONPATH=$PYTHONPATH:$(realpath ./src)
        fi
      '';
    };
  };
}
