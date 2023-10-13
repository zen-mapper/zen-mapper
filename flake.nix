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

    packages.${system}.default = pkgs.python3Packages.buildPythonPackage {
      pname = "zen-mapper";
      version = "0.1.0";
      format = "pyproject";

      src = ./.;

      nativeBuildInputs = [
        pkgs.hatch
      ];
    };

    devShells.${system}.default = pkgs.mkShell {
      buildInputs = with pkgs; [
        rye
      ];

      shellHook = ''
        [ -d .venv ] || rye sync
        . .venv/bin/activate
      '';
    };
  };
}
