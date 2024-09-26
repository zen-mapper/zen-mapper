{
  buildPythonPackage,
  pythonOlder,
  fetchFromGitHub,
  # Build
  setuptools,
  setuptools-scm,
  # Dependencies
  pillow,
  sphinx,
  # Check
  lxml,
  matplotlib,
  numpy,
  pytestCheckHook,
  pytestcov,
}:
buildPythonPackage rec {
  pname = "sphinx-gallery";
  version = "0.17.1";
  pyproject = true;

  disabled = pythonOlder "3.8";

  src = fetchFromGitHub {
    owner = "sphinx-gallery";
    repo = pname;
    rev = "v${version}";
    hash = "sha256-Xk1O+xnuG3TBfFooQwERzSveAix3MipW0lqrzU+vpaI=";
  };

  dependencies = [
    pillow
    sphinx
  ];

  build-system = [
    setuptools
    setuptools-scm
  ];

  disabledTests = [
    "test_dummy_image"
    "test_embed_code_links_get_data"
  ];

  nativeCheckInputs = [
    lxml
    matplotlib
    numpy
    pytestCheckHook
    pytestcov
  ];
}
