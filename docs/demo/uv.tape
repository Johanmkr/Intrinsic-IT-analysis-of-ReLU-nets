# Clone and run with uv. Record from the repository root:  vhs docs/demo/uv.tape
# (set DEMO_REPO=$PWD to clone the local checkout instead of GitHub).
# Long steps run hidden; the recording resumes when the prompt returns.
Output docs/demo/uv.gif
Source docs/demo/common.tape

Type "git clone https://github.com/Johanmkr/Intrinsic_IT_analysis_of_ReLU_nets.git" Enter
Wait@2m /^\$\s*$/
Type "cd Intrinsic_IT_analysis_of_ReLU_nets" Enter
Sleep 500ms

Type "# environment (uv) + MNIST and WBC" Enter
Type "./run.sh setup" Enter
Sleep 3s
Hide
Wait@10m /^\$\s*$/
Show
Sleep 2s

Type "# A: redraw every figure from the stored results (~40 s)" Enter
Type "./run.sh step5" Enter
Sleep 3s
Hide
Wait@10m /^\$\s*$/
Show
Sleep 1s
Type "ls figures/*.pdf" Enter
Sleep 3s

Type "clear" Enter
Type "# B: rerun every experiment from scratch: ./run.sh all (~2 h)" Enter
Type "# here: the whole pipeline on a tiny sweep (~5 min)" Enter
Type "./run.sh smoke" Enter
Sleep 6s
Hide
Wait@20m /^\$\s*$/
Show
Sleep 4s
