{
  lib,
  buildPythonPackage,
  pythonOlder,
  # Build
  hatchling,
  # Dependencies
  numpy,
  typing-extensions,
  zen-mapper,
  # Check
  pytestCheckHook,
  scikit-learn,
  hypothesis,
  networkx,
}:
buildPythonPackage {
  pname = "kaiju-mapper";
  version = "0.1.1";
  pyproject = true;

  disabled = pythonOlder "3.10";

  src = ../packages/kaiju-mapper;

  dependencies = [
    numpy
    scikit-learn
    zen-mapper
  ]
  ++ lib.optionals (pythonOlder "3.11") [ typing-extensions ];

  build-system = [
    hatchling
  ];

  nativeCheckInputs = [
    pytestCheckHook
    hypothesis
  ];
}
