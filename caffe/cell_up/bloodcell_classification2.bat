D:
cd D:\caffeDev
cd caffe-master

set TOOLS=Build\x64\Release
set EXAMPLE=examples\bloodcell
set DATA=data\bloodcell

%TOOLS%\classification %EXAMPLE%\bloodcell_classification2.prototxt %EXAMPLE%\bloodcell_full2\_iter_10000.caffemodel.h5 %EXAMPLE%\bloodcell_train_mean.binaryproto %DATA%/class.txt %DATA%\test.jpeg

pause