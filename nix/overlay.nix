final: prev: {
  pythonPackagesExtensions =
    (prev.pythonPackagesExtensions or [])
    ++ [
      (python-final: python-prev: {
        zen-mapper = python-final.callPackage ./zen-mapper.nix {};
      })
    ];
}
