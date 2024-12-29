{
  lib,
  buildPythonPackage,
  pythonOlder,
  # Build
  hatchling,
  # Dependencies
  numpy,
  typing-extensions,
  # Check
  pytestCheckHook,
  scikit-learn,
  hypothesis,
  networkx,
}:
buildPythonPackage {
  pname = "zen-mapper";
  version = "0.2.0";
  pyproject = true;

  disabled = pythonOlder "3.10";

  src = ../.;

  dependencies = [
    numpy
  ] ++ lib.optionals (pythonOlder "3.11") [ typing-extensions ];

  build-system = [
    hatchling
  ];

  nativeCheckInputs = [
    pytestCheckHook
    hypothesis
    scikit-learn
    networkx
  ];
}
