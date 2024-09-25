{
  buildPythonPackage,
  pythonOlder,
  # Build
  hatchling,
  # Dependencies
  numpy,
  # Check
  pytestCheckHook,
  scikit-learn,
  hypothesis,
  networkx,
}:
buildPythonPackage {
  pname = "zen-mapper";
  version = "0.1.4";
  format = "pyproject";

  disabled = pythonOlder "3.11";

  src = ../.;

  propagatedBuildInputs = [
    numpy
  ];

  nativeBuildInputs = [
    hatchling
  ];

  nativeCheckInputs = [
    pytestCheckHook
    hypothesis
    scikit-learn
    networkx
  ];
}
