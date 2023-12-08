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
    python = pkgs.python311.withPackages (ps: [
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

    packages.${system}.default = pkgs.python311Packages.callPackage ./nix/zen-mapper.nix {};

    devShells.${system}.default = pkgs.mkShell {
      venvDir = ".venv";
      buildInputs = [
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
