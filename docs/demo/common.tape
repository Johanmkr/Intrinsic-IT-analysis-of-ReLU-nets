# Shared settings for the demo GIFs (sourced by uv.tape and docker.tape).
Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 680
Set Padding 20
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set TypingSpeed 35ms

# Hidden: plain prompt, empty working directory, and (until the repository is
# public) clone from the local checkout $DEMO_REPO instead of GitHub.
Hide
Type `export PS1='$ ' DEMO_TMP=$(mktemp -d) && cd "$DEMO_TMP"` Enter
Type `[ -n "$DEMO_REPO" ] && export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.$DEMO_REPO.insteadOf" GIT_CONFIG_VALUE_0=https://github.com/Johanmkr/Intrinsic_IT_analysis_of_ReLU_nets.git` Enter
Type "clear" Enter
Show
