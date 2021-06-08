import unittest
import numpy as np
from torch import zeros_like

from nn_layers import *
from models import *
from utils import *


class TestMobiusConv(unittest.TestCase):
    """Unit tests for class MobiusConv"""

    def test_expand_kernel(self):
        """Checking kernel symmetries after expansion from parameter array"""
        mobius_conv = MobiusConv(in_fields=(1,1,1), out_fields=(1,1,1), kernel_size=5)
        kernel = mobius_conv._expand_kernel()
        k_triv2triv = kernel[0,0]
        self.assertTrue(torch.allclose(k_triv2triv, k_triv2triv.flip(0)))
        k_sign2sign = kernel[1,1]
        self.assertTrue(torch.allclose(k_sign2sign, k_sign2sign.flip(0)))
        k_triv2sign = kernel[1,0]
        self.assertTrue(torch.allclose(k_triv2sign, -k_triv2sign.flip(0)))
        k_sign2triv = kernel[0,1]
        self.assertTrue(torch.allclose(k_sign2triv, -k_sign2triv.flip(0)))
        k_triv2reg_1 = kernel[2,0]
        k_triv2reg_2 = kernel[3,0]
        self.assertTrue(torch.allclose(k_triv2reg_1, k_triv2reg_2.flip(0)))
        k_sign2reg_1 = kernel[2,1]
        k_sign2reg_2 = kernel[3,1]
        self.assertTrue(torch.allclose(k_sign2reg_1, -k_sign2reg_2.flip(0)))
        k_reg2triv_1 = kernel[0,2]
        k_reg2triv_2 = kernel[0,3]
        self.assertTrue(torch.allclose(k_reg2triv_1, k_reg2triv_2.flip(0)))
        k_reg2sign_1 = kernel[1,2]
        k_reg2sign_2 = kernel[1,3]
        self.assertTrue(torch.allclose(k_reg2sign_1, -k_reg2sign_2.flip(0)))
        k_reg2reg_11 = kernel[2,2]
        k_reg2reg_12 = kernel[2,3]
        k_reg2reg_21 = kernel[3,2]
        k_reg2reg_22 = kernel[3,3]
        self.assertTrue(torch.allclose(k_reg2reg_11, k_reg2reg_22.flip(0)))
        self.assertTrue(torch.allclose(k_reg2reg_12, k_reg2reg_21.flip(0)))

    def test_expand_bias(self):
        """Checking bias symmetries after expansion from parameter array"""
        mobius_conv = MobiusConv(in_fields=(1,1,1), out_fields=(1,1,1), kernel_size=5)
        # set random bias parameters since they are by default initialized to zero
        mobius_conv.bias_triv.data = torch.randn_like(mobius_conv.bias_triv)
        mobius_conv.bias_reg.data  = torch.randn_like(mobius_conv.bias_reg)
        bias = mobius_conv._expand_bias()
        self.assertTrue(bias[1]==0)
        self.assertTrue(bias[2]==bias[3])

    def test_transport_pad(self):
        """Checking transport padding operation"""
        size = 5
        mobius_conv = MobiusConv(in_fields=(1,1,1), out_fields=(1,1,1), kernel_size=size)
        N,C,W,L = 2,4,8,12
        input = torch.randn(N,C,W,L)
        padded = mobius_conv._transport_pad(input)
        # top and bottom rows should be padded with zeros
        self.assertTrue(np.allclose(padded[:,:,:size//2,:].detach().numpy(), 0))
        self.assertTrue(np.allclose(padded[:,:,-(size//2):,:].detach().numpy(), 0))
        # test scalar field transport
        self.assertTrue(torch.allclose(
            padded[:, 0:1, :, :size//2].flip(2),
            padded[:, 0:1, :, -2*(size//2):-(size//2)]))
        self.assertTrue(torch.allclose(
            padded[:, 0:1, :, size//2:2*(size//2)].flip(2),
            padded[:, 0:1, :, -(size//2):]))
        # test signflip field transport
        self.assertTrue(torch.allclose(
            padded[:, 1:2, :, :size//2].flip(2),
            - padded[:, 1:2, :, -2*(size//2):-(size//2)]))
        self.assertTrue(torch.allclose(
            padded[:, 1:2, :, size//2:2*(size//2)].flip(2),
            - padded[:, 1:2, :, -(size//2):]))
        # test regular feature field transport
        self.assertTrue(torch.allclose(
            padded[:, -2:, :, :size//2].flip(2).flip(1),
            padded[:, -2:, :, -2*(size//2):-(size//2)]))
        self.assertTrue(torch.allclose(
            padded[:, -2:, :, size//2:2*(size//2)].flip(2).flip(1),
            padded[:, -2:, :, -(size//2):]))

    def test_forward_gauge_equivariance(self):
        """Checking gauge equivariance of the forward pass (w.r.t. global flip of gauges)"""
        mobius_conv = MobiusConv(in_fields=(1,1,1), out_fields=(1,1,1), kernel_size=5).double()
        # set random bias parameters since they are by default initialized to zero
        mobius_conv.bias_triv.data = torch.randn_like(mobius_conv.bias_triv)
        mobius_conv.bias_reg.data  = torch.randn_like(mobius_conv.bias_reg)
        N,C,W,L = 2,4,11,17
        # first convolve, then gauge transform
        input = torch.randn(N,C,W,L).double()
        output = mobius_conv.forward(input)
        output_conv_flip = gauge_transform(output, (1,1,1))
        # first gauge transform, then convolve
        mobius_conv._reflect_kernel() # align kernels relative to reflected frames
        input_flip = gauge_transform(input, (1,1,1))
        output_flip_conv = mobius_conv.forward(input_flip)
        self.assertTrue(torch.allclose(output_flip_conv, output_conv_flip))

    def test_forward_isometry_equivariance(self):
        """Checking isometry equivariance of the forward pass w.r.t. arbitrary shifts"""
        mobius_conv = MobiusConv(in_fields=(1,1,1), out_fields=(1,1,1), kernel_size=5).double()
        # set random bias parameters since they are by default initialized to zero
        mobius_conv.bias_triv.data = torch.randn_like(mobius_conv.bias_triv)
        mobius_conv.bias_reg.data  = torch.randn_like(mobius_conv.bias_reg)
        N,C,W,L = 2,4,11,17
        input = torch.randn(N,C,W,L).double()
        output = mobius_conv.forward(input)
        for shift in range(2*L+1):
            input_shift = isom_action(input, shift, (1,1,1))
            output_shift_conv = mobius_conv.forward(input_shift)
            output_conv_shift = isom_action(output, shift, (1,1,1))
            self.assertTrue(torch.allclose(output_shift_conv, output_conv_shift))




class TestEquivNonlin(unittest.TestCase):
    """Unit tests for class EquivNonlin"""

    def test_forward_gauge_equivariance(self):
        """Checking gauge equivariance of the forward pass (w.r.t. global flip of gauges)"""
        equiv_nonlin = EquivNonlin(in_fields=(1,1,1)).double()
        equiv_nonlin.bias_sign.data = torch.randn_like(equiv_nonlin.bias_sign) # set random bias parameters
        N,C,W,L = 2,4,11,17
        # first apply nonlinearity, then gauge transform
        input = torch.randn(N,C,W,L).double()
        output = equiv_nonlin.forward(input)
        output_nonlin_flip = gauge_transform(output, (1,1,1))
        # first gauge transform, then apply nonlinearity
        input_flip = gauge_transform(input, (1,1,1))
        output_flip_nonlin = equiv_nonlin.forward(input_flip)
        self.assertTrue(torch.allclose(output_flip_nonlin, output_nonlin_flip))

    def test_forward_isometry_equivariance(self):
        """Checking isometry equivariance of the forward pass w.r.t. arbitrary shifts"""
        equiv_nonlin = EquivNonlin(in_fields=(1,1,1)).double()
        equiv_nonlin.bias_sign.data = torch.randn_like(equiv_nonlin.bias_sign) # set random bias parameters
        N,C,W,L = 2,4,11,17
        input = torch.randn(N,C,W,L).double()
        output = equiv_nonlin.forward(input)
        for shift in range(2*L+1):
            input_shift = isom_action(input, shift, (1,1,1))
            output_shift_nonlin = equiv_nonlin.forward(input_shift)
            output_nonlin_shift = isom_action(output, shift, (1,1,1))
            self.assertTrue(torch.allclose(output_shift_nonlin, output_nonlin_shift))




class TestMobiusPool(unittest.TestCase):
    """Unit tests for class MobiusPool"""

    def test_forward_gauge_equivariance(self):
        """Checking gauge equivariance of the forward pass (w.r.t. global flip of gauges)"""
        mobius_pool = MobiusPool(in_fields=(1,1,1)).double()
        N,C,W,L = 2,4,11,17
        # first pool, then gauge transform
        input = torch.randn(N,C,W,L).double()
        output = mobius_pool.forward(input)
        output_pool_flip = gauge_transform(output, (1,1,1))
        # first gauge transform, then pool
        input_flip = gauge_transform(input, (1,1,1))
        output_flip_pool = mobius_pool.forward(input_flip)
        self.assertTrue(torch.allclose(output_flip_pool, output_pool_flip))

    def test_forward_isometry_equivariance(self):
        """Checking isometry equivariance of the forward pass

        Due to the discretization, isometry equivariance holds only for the subgroup of isometries which shifts by an
        even number of pixels. The strip is furthermore required to have an even length.
        Note that the same issues apply for conventional Euclidean convolutions.
        Our empirical evaluation in Table 5 of the paper shows that isometry equivariance does for arbitrary shifts
        still hold approximately.
        """
        mobius_pool = MobiusPool(in_fields=(1,1,1)).double()
        N,C,W,L = 2,4,11,18 # even length of the strip
        input = torch.randn(N,C,W,L).double()
        output = mobius_pool.forward(input)
        for shift in range(0, 2*L+3, 2): # shift by an even number of pixels
            input_shift = isom_action(input, shift, (1,1,1))
            output_shift_pool = mobius_pool.forward(input_shift)
            output_pool_shift = isom_action(output, shift//2, (1,1,1))
            self.assertTrue(torch.allclose(output_shift_pool, output_pool_shift))




class TestMobiusGaugeCNN(unittest.TestCase):
    """Unit tests for class MobiusGaugeCNN"""

    def test_forward_gauge_equivariance(self):
        """Checking gauge invariance of all orientation independent models (w.r.t. global flip of gauges)"""
        for mode in ('scalar', 'signflip', 'regular', 'mixed', 'irrep'):
            result = self._test_forward_gauge_equivariance(mode)
            self.assertTrue(result)
        
    def _test_forward_gauge_equivariance(self, mode):
        """Checking gauge invariance of the forward pass of the model specified by the argument 'mode'
        ((w.r.t. global flip of gauges)"""
        model = MobiusGaugeCNN(kernel_size=5, ch_in=3, classes=10, mode=mode)
        model.eval() # deterministic
        model = model.double()
        # generate random image (interpreted as three scalar fields)
        img_orig = torch.randn((2, 3, 28, 28), dtype=torch.double)
        out_orig = model(img_orig)
        img_flip = gauge_transform(img_orig, in_fields=(3,0,0))
        out_flip = model(img_flip)
        return torch.allclose(out_orig, out_flip)

    def test_forward_isometry_equivariance(self):
        """Checking isometry invariance of all orientation independent models"""
        for mode in ('scalar', 'signflip', 'regular', 'mixed', 'irrep'):
            result = self._test_forward_isometry_equivariance(mode)
            self.assertTrue(result)

    def _test_forward_isometry_equivariance(self, mode):
        """Checking isometry invariance of the forward pass of the model specified by the argument 'mode'

        Due to the discretization and the use of two pooling layers, isometry equivariance holds only for the subgroup
        of isometries which shifts by multiples of 4 pixels. The strip is furthermore required to have a length which is
        divisible by 4.
        Note that the same issues apply for conventional Euclidean CNNs.
        Our empirical evaluation in Table 5 of the paper shows that isometry equivariance does for arbitrary shifts
        still hold approximately.
        """
        model = MobiusGaugeCNN(kernel_size=5, ch_in=3, classes=10, mode='mixed') # mode='mixed' uses all field types
        model.eval() # deterministic
        model = model.double()
        # generate random image (interpreted as three scalar fields)
        img_orig = torch.randn((2, 3, 28, 28), dtype=torch.double)
        out_orig = model(img_orig)
        for shift in range(0, 2*28+5, 4): # shift by multiples of 4 pixels
            img_shift = isom_action(img_orig, shift, in_fields=(3,0,0))
            out_shift = model.forward(img_shift)
            return torch.allclose(out_orig, out_shift)



if __name__ == '__main__':
    unittest.main()
