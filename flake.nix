{
  description = "Lightweight Japanese to Chinese subtitle pipeline";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.url = "github:pyproject-nix/uv2nix";
    uv2nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs";
    pyproject-build-systems.inputs.nixpkgs.follows = "nixpkgs";
    pyproject-build-systems.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.inputs.uv2nix.follows = "uv2nix";
    audio-transcribe.url = "github:nemowang2003/audio-transcribe";
    audio-transcribe.inputs.nixpkgs.follows = "nixpkgs";
    audio-transcribe.inputs.pyproject-nix.follows = "pyproject-nix";
    audio-transcribe.inputs.uv2nix.follows = "uv2nix";
    audio-transcribe.inputs.pyproject-build-systems.follows = "pyproject-build-systems";
  };
  outputs = {
    self,
    nixpkgs,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
    audio-transcribe,
    ...
  }: let
    systems = ["x86_64-linux" "aarch64-linux" "aarch64-darwin"];
    each-system = nixpkgs.lib.genAttrs systems;
    workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = self;};
    overlay = workspace.mkPyprojectOverlay {sourcePreference = "wheel";};
    mk-application = pkgs: (pkgs.callPackages pyproject-nix.build.util {}).mkApplication;
  in {
    lib.runtimeOverlay = _: final: prev: {
      srt = prev.srt.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ final.resolveBuildSystem {setuptools = [];};
      });
    };
    lib.mkPythonSet = {
      pkgs,
      python ? pkgs.python314,
    }:
      (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (pkgs.lib.composeManyExtensions [
        pyproject-build-systems.overlays.wheel
        overlay
        (self.lib.runtimeOverlay {inherit pkgs;})
        (final: prev:
          pkgs.lib.genAttrs ["ja2zh-srt" "audio-transcribe"] (name:
            prev.${name}.overrideAttrs (old: {
              # Use the backend from uv.lock instead of uv's bundled version.
              uvBuildFlags = (old.uvBuildFlags or []) ++ ["--force-pep517"];
              nativeBuildInputs = (old.nativeBuildInputs or []) ++ final.resolveBuildSystem {uv-build = [];};
            })))
      ]);
    lib.mkPackage = {
      pkgs,
      cudaSupport ? false,
      python ? pkgs.python314,
    }: let
      python-set = self.lib.mkPythonSet {inherit pkgs python;};
      environment = python-set.mkVirtualEnv "ja2zh-srt-env" {ja2zh-srt = [];};
      app = mk-application pkgs {
        venv = environment;
        package = python-set.ja2zh-srt;
      };
      cuda-library-path = (audio-transcribe.lib.mkPackage {inherit pkgs python;}).cuda-library-path;
    in
      app.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pkgs.makeWrapper];
        postFixup =
          (old.postFixup or "")
          + pkgs.lib.optionalString cudaSupport ''
            wrapProgram "$out/bin/ja2zh-srt" --prefix LD_LIBRARY_PATH : "${cuda-library-path}:/run/opengl-driver/lib:/usr/lib/wsl/lib"
          '';
        meta.mainProgram = "ja2zh-srt";
        passthru =
          (old.passthru or {})
          // {
            python-env = environment;
            tests.unit = pkgs.runCommand "ja2zh-srt-unit" {} ''
              export PYTHONDONTWRITEBYTECODE=1
              ${environment}/bin/python -m unittest discover -s ${self}/tests
              touch "$out"
            '';
            tests.audio-transcribe = pkgs.runCommand "ja2zh-srt-audio-transcribe" {} ''
              export PYTHONDONTWRITEBYTECODE=1
              ${environment}/bin/python -m unittest discover -s ${audio-transcribe}/tests
              ${environment}/bin/python -c 'import audio_transcribe, av, ctranslate2, faster_whisper, onnxruntime'
              touch "$out"
            '';
          };
      });
    packages = each-system (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
      {
        default = self.lib.mkPackage {inherit pkgs;};
        ja2zh-srt = self.packages.${system}.default;
      }
      // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
        cuda = self.lib.mkPackage {
          inherit pkgs;
          cudaSupport = true;
        };
      });
    checks = each-system (system: self.packages.${system}.default.tests);
    devShells = each-system (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      default = pkgs.mkShell {
        packages = [pkgs.uv pkgs.python314];
        UV_PYTHON = pkgs.python314.interpreter;
        UV_PYTHON_DOWNLOADS = "never";
      };
    });
    formatter = each-system (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
