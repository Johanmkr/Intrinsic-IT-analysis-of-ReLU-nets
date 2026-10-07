# Clone and run with Docker. Record from the repository root:  vhs docs/demo/docker.tape
# (set DEMO_REPO=$PWD to clone the local checkout instead of GitHub).
# Long steps run hidden; the recording resumes when the prompt returns.
Output docs/demo/docker.gif
Source docs/demo/common.tape

Type "git clone https://github.com/Johanmkr/Intrinsic_IT_analysis_of_ReLU_nets.git" Enter
Wait@2m /^\$\s*$/
Type "cd Intrinsic_IT_analysis_of_ReLU_nets" Enter
Sleep 500ms

Type "# image: locked environment, MNIST, WBC and the stored results" Enter
Type "docker build -t intrinsic-it-relu-nets ." Enter
Sleep 4s
Hide
Wait@20m /^\$\s*$/
Show
Sleep 2s

Type "clear" Enter
Type "# A: redraw every figure from the stored results" Enter
Type "mkdir -p figures" Enter
Type `docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/figures:/app/figures" \` Enter
Type "    intrinsic-it-relu-nets step5" Enter
Sleep 3s
Hide
Wait@10m /^\$\s*$/
Show
Sleep 1s
Type "ls figures/*.pdf" Enter
Sleep 3s

Type "clear" Enter
Type "# B: rerun every experiment from scratch: ... intrinsic-it-relu-nets all (~2 h)" Enter
Type "# here: the whole pipeline on a tiny sweep (~5 min)" Enter
Type "docker run --rm intrinsic-it-relu-nets smoke --workers 8" Enter
Sleep 6s
Hide
Wait@20m /^\$\s*$/
Show
Sleep 4s
