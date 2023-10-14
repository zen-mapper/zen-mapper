{
  buildPythonPackage,
  numpy,
  hatchling,
  pytest,
  hypothesis,
}:
buildPythonPackage {
  pname = "zen-mapper";
  version = "0.1.0";
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
  ];

  checkPhase = ''
    runHook preCheck

    pytest

    runHook postCheck
  '';
}
