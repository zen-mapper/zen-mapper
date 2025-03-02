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
  in {
    formatter.${system} = pkgs.alejandra;

    packages.${system}.default = pkgs.python3Packages.callPackage ./nix/zen-mapper.nix {};

    overlays.default = import ./nix/overlay.nix;

    templates.default = {
      path = ./nix/templates/minimal;
      description = "Minimal example of using zen-mapper with flakes";
    };

    checks.${system} = builtins.listToAttrs (
      map
      (python: {
        name = python;
        value = pkgs."${python}".pkgs.callPackage ./nix/zen-mapper.nix {};
      })
      [
        "python313"
        "python312"
        "python311"
        "python310"
      ]
    );

    devShells.${system}.default = pkgs.mkShell {
      NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc
        pkgs.libz
      ];

      LC_ALL = "en_US.UTF-8";

      buildInputs = [
        pkgs.uv
        pkgs.hatch
        pkgs.jq
        pkgs.just
        pkgs.ruff
        self.formatter.${system}
      ];

      shellHook = ''
        if [ -z ''${NIX_LD+x} ]
        then
          export LD_LIBRARY_PATH="$NIX_LD_LIBRARY_PATH"
        fi
        uv sync --group docs --group dev
        source .venv/bin/activate
      '';
    };
  };
}
