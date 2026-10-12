{
  description = "Lightweight Japanese to Chinese subtitle pipeline";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    audio-transcribe.url = "github:nemowang2003/audio-transcribe";
    audio-transcribe.inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs = {
    self,
    nixpkgs,
    audio-transcribe,
  }: let
    systems = ["x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin"];
    each-system = nixpkgs.lib.genAttrs systems;
  in {
    lib.mkPackage = {
      pkgs,
      cudaSupport ? false,
      python ? pkgs.python314,
    }: let
      sdk = audio-transcribe.lib.mkPackage {inherit pkgs python cudaSupport;};
      app = python.pkgs.buildPythonApplication {
        pname = "ja2zh-srt";
        version = "0.2.0";
        src = self;
        pyproject = true;
        build-system = [python.pkgs.uv-build];
        dependencies = [sdk python.pkgs.click python.pkgs.httpx python.pkgs.openai python.pkgs.srt];
        pythonImportsCheck = ["ja2zh_srt"];
        nativeBuildInputs = pkgs.lib.optionals cudaSupport [pkgs.makeWrapper];
        postFixup = pkgs.lib.optionalString cudaSupport ''
          wrapProgram "$out/bin/ja2zh-srt" --prefix LD_LIBRARY_PATH : "${sdk.cuda-library-path}:/run/opengl-driver/lib:/usr/lib/wsl/lib"
        '';
        passthru.tests.unit = pkgs.runCommand "ja2zh-srt-unit" {} ''
          export PYTHONPATH=${self}/src:${audio-transcribe}/src
          export PYTHONDONTWRITEBYTECODE=1
          ${python.withPackages (ps: [ps.click ps.httpx ps.openai ps.srt])}/bin/python -m unittest discover -s ${self}/tests
          touch "$out"
        '';
      };
    in
      app;
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
    checks = each-system (system: {unit = self.packages.${system}.default.tests.unit;});
    devShells = each-system (system: let pkgs = nixpkgs.legacyPackages.${system}; in {default = pkgs.mkShell {packages = [pkgs.uv pkgs.python314];};});
    formatter = each-system (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
