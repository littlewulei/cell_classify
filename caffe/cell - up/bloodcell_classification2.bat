D:
cd D:\caffeDev
cd caffe-master

set TOOLS=Build\x64\Release
set EXAMPLE=examples\cell
set DATA=data\cell

%TOOLS%\classification %EXAMPLE%\Alexnet_resize_classification.prototxt %EXAMPLE%\cell_model_resize\_iter_10000.caffemodel %DATA%\cell_train_mean_lmdb1.binaryproto %DATA%/class.txt %DATA%\test.jpeg

pause