D:
cd D:\caffeDev
cd caffe-master

set TOOLS=Build\x64\Release
set EXAMPLE=examples\bloodcell
set DATA=data\bloodcell

%TOOLS%\caffe train --solver=%EXAMPLE%\bloodcell_full_solver2.prototxt --snapshot=%EXAMPLE%\bloodcell_full2\_iter_9500.solverstate.h5

pause