nix-shell -E 'with import <nixpkgs> {}; (pkgs.buildFHSUserEnv { name = "fhs"; }).env'
