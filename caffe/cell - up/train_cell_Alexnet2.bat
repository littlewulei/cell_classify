D:
cd D:\caffeDev\caffe-master

set GLOG_logtostderr=1
set BIN=Build/x64/Release

"%BIN%/caffe.exe" train --solver=examples/cell/cell_solver-leveldb2.prototxt 
pause