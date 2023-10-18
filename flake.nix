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

    packages.${system}.default = pkgs.python311Packages.callPackage ./nix/zen-mapper.nix {};

    devShells.${system}.default = pkgs.mkShell {
      buildInputs = with pkgs; [
        act
        rye
      ];

      shellHook = ''
        [ -d .venv ] && .venv/bin/activate
      '';
    };
  };
}
