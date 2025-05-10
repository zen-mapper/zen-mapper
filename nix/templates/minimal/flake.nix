{
  description = "A minimal flake loading zen-mapper";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";

    zen-mapper = {
      url = "github:zen-mapper/zen-mapper";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    zen-mapper,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system}.appendOverlays [zen-mapper.overlays.default];
    python = pkgs.python3;
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        (python.withPackages (ps: [ps.zen-mapper]))
      ];
    };
  };
}
