{
  # Build
  buildPythonPackage,
  hatchling,

  # Dependencies
  numpy,

  # Check
  pytest,
  scikit-learn,
  hypothesis,
}:
buildPythonPackage {
  pname = "zen-mapper";
  version = "0.1.3";
  format = "pyproject";

  src = ../.;

  propagatedBuildInputs = [
    numpy
  ];

  nativeBuildInputs = [
    hatchling
  ];

  nativeCheckInputs = [
    pytest
    hypothesis
    scikit-learn
  ];

  checkPhase = ''
    runHook preCheck

    pytest

    runHook postCheck
  '';
}
