{
  description = "CUDA-only Japanese MP3 to Japanese/Chinese SRT pipeline";

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    devshell.url = "github:numtide/devshell";
    devshell.inputs.nixpkgs.follows = "nixpkgs";

    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";

    uv2nix.url = "github:pyproject-nix/uv2nix";
    uv2nix.inputs.pyproject-nix.follows = "pyproject-nix";
    uv2nix.inputs.nixpkgs.follows = "nixpkgs";

    pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs";
    pyproject-build-systems.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.inputs.uv2nix.follows = "uv2nix";
    pyproject-build-systems.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    devshell,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
    ...
  }: let
    inherit (nixpkgs) lib;
    systems = ["x86_64-linux"];
    forAllSystems = lib.genAttrs systems;
    mkSystem = system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [devshell.overlays.default];
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      python = pkgs.python312;
      workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};
      projectOverlay = workspace.mkPyprojectOverlay {sourcePreference = "wheel";};
      pyprojectOverrides = final: prev: {
        srt = prev.srt.overrideAttrs (old: {
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.setuptools];
        });
      };
      pythonSet = (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          projectOverlay
          pyprojectOverrides
        ]
      );

      pythonEnv = pythonSet.mkVirtualEnv "ja2zh-srt-python-env" workspace.deps.default;
      llamaCppCuda = pkgs.llama-cpp;
      runtimeTools = [
        pkgs.ffmpeg
        llamaCppCuda
      ];

      app = pkgs.symlinkJoin {
        name = "ja2zh-srt";
        paths = [
          pythonEnv
          pkgs.ffmpeg
          llamaCppCuda
        ];
        nativeBuildInputs = [pkgs.makeWrapper];
        postBuild = ''
          wrapProgram "$out/bin/ja2zh-srt" \
            --prefix PATH : "${lib.makeBinPath runtimeTools}"
        '';
      };
    in {
      inherit
        app
        llamaCppCuda
        pkgs
        python
        pythonEnv
        ;
    };
  in {
    packages = forAllSystems (
      system: let
        env = mkSystem system;
      in {
        default = env.app;
        ja2zh-srt = env.app;
        python-env = env.pythonEnv;
        llama-cpp-cuda = env.llamaCppCuda;
      }
    );

    apps = forAllSystems (
      system: let
        env = mkSystem system;
      in {
        default = {
          type = "app";
          program = "${env.app}/bin/ja2zh-srt";
        };
        ja2zh-srt = {
          type = "app";
          program = "${env.app}/bin/ja2zh-srt";
        };
      }
    );

    devShells = forAllSystems (
      system: let
        env = mkSystem system;
      in {
        # 💡 现在可以直接调用被注入的 pkgs.devshell.mkShell 了！
        default = env.pkgs.devshell.mkShell {
          name = "ja2zh-srt-cuda-env";

          # 告别 bash 的 export/unset，纯结构化声明
          env = [
            {
              name = "PYTHONPATH";
              unset = true;
            }
            {
              name = "UV_NO_SYNC";
              value = "1";
            }
            {
              name = "UV_PYTHON_DOWNLOADS";
              value = "never";
            }
          ];

          packages = [
            env.pythonEnv
            env.pkgs.uv
            env.pkgs.ffmpeg
            env.llamaCppCuda
          ];
        };
      }
    );

    checks = forAllSystems (
      system: let
        env = mkSystem system;
      in {
        help = env.pkgs.runCommand "ja2zh-srt-help" {nativeBuildInputs = [env.app];} ''
          ja2zh-srt --help > "$out"
        '';
      }
    );

    formatter = forAllSystems (system: (mkSystem system).pkgs.nixfmt);
  };
}
