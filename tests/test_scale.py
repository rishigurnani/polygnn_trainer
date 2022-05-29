from polygnn_trainer.scale import *
import numpy as np
from torch import tensor
from torch import float as torch_float


def test_basic():
    data = np.array([[100], [0.01], [1]])
    MyScaler = SequentialScaler()
    MyScaler.append(LogTenScaler())
    MyScaler.append(MinMaxScaler())
    data_trans = MyScaler.fit_transform(data)
    out = MyScaler.inverse_transform(data_trans)
    assert (out == tensor(data, dtype=torch_float)).all()


def test_transform():

    data = np.array([[100], [0.01], [1]])
    MyScaler = SequentialScaler()
    MyScaler.append(LogTenScaler())
    MyScaler.append(MinMaxScaler())
    data_fittrans = MyScaler.fit_transform(data)
    data_trans = MyScaler.transform(data)
    assert (data_fittrans == data_trans).all()


def test_transform_zeromean():

    data = np.array([[100], [0.01], [1]])
    MyScaler = SequentialScaler()
    MyScaler.append(ZeroMeanScaler())
    MyScaler.append(MinMaxScaler())
    data_fittrans = MyScaler.fit_transform(data)
    data_trans = MyScaler.transform(data)
    assert (data_fittrans == data_trans).all()


def test_quotient_scaler():

    data = tensor([[10], [8]], dtype=torch_float)
    MyScaler = SequentialScaler()
    MyScaler.append(QuotientScaler(2))
    data_scale = MyScaler.fit_transform(data)
    assert (data_scale == tensor([[5], [4]], dtype=torch_float)).all()

    data_unscale = MyScaler.inverse_transform(data_scale)

    assert (data_unscale == data).all()


def test_product_scaler():

    data = tensor([[10], [8]], dtype=torch_float)
    MyScaler = SequentialScaler()
    MyScaler.append(ProductScaler(2))
    data_scale = MyScaler.fit_transform(data)
    assert (data_scale == tensor([[20], [16]], dtype=torch_float)).all()

    data_unscale = MyScaler.inverse_transform(data_scale)

    assert (data_unscale == data).all()


def test_islinear():

    MyScaler = SequentialScaler()
    MyScaler.append(LogTenScaler())
    MyScaler.append(MinMaxScaler())
    assert MyScaler.is_linear() == False

    MyScaler = SequentialScaler()
    MyScaler.append(MinMaxScaler())
    assert MyScaler.is_linear() == True


def test_string():

    data = np.array([[100], [0.01], [1]])
    MyScaler = SequentialScaler()
    MyScaler.append(LogTenScaler())
    MyScaler.append(MinMaxScaler())
    MyScaler.fit_transform(data)
    assert (
        str(MyScaler) == "Forward: LogTenScaler --> MinMaxScaler(max: 2.0, min: -2.0)"
    )
