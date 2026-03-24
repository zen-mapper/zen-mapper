final: prev: {
  pythonPackagesExtensions =
    (prev.pythonPackagesExtensions or [])
    ++ [
      (python-final: python-prev: {
        zen-mapper = python-final.callPackage ./zen-mapper.nix {};
      })
      (python-final: python-prev: {
        kaiju-mapper = python-final.callPackage ./kaiju-mapper.nix {};
      })
    ];
}
