class Recallforge < Formula
  include Language::Python::Virtualenv

  desc "Local multimodal memory for AI agents"
  homepage "https://github.com/brianmeyer/recallforge"
  url "https://pypi.org/packages/source/r/recallforge/recallforge-0.2.0.tar.gz"
  # sha256 will be filled after PyPI publish
  license "MIT"

  depends_on "python@3.13"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "RecallForge", shell_output("#{bin}/recallforge --version")
  end
end
