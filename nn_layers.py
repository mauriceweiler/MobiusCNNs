import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MobiusConv(nn.Module):
    """Orientation independent convolution layer on the Mobius strip

    Implementation of the orientation independent convolutions and bias summation as described in Sections 5.3 and 5.4.
    Three different field types, which transform according to the trivial, signflip and regular representation of the
    reflection group G=R, are implemented. The convolution is computed by:
        1) parallel transport padding the input field (see Fig. 29)
        2) expanding reflection steerable kernels (see Table 3 and Section 5.3.3)
        3) expanding reflection steerable biases (see Section 5.3.1)
        4) performing a conventional Euclidean convolution with the steerable
           kernels and summing the expanded biases

    """
    def __init__(self,
                 in_fields,
                 out_fields,
                 kernel_size):
        """
        Args:
            in_fields (Tuple[int, int, int]): multiplicities of the input feature fields, transforming according to
                trivial, signflip and regular representations, respectively.
            out_fields (Tuple[int, int, int]): multiplicities of the output feature fields, transforming according to
                trivial, signflip and regular representations, respectively.
            kernel_size (int): size (height, width) of the convolution kernel

        """
        super(MobiusConv, self).__init__()
        
        assert kernel_size%2 == 1 # only odd kernel sizes allowed due to padding
        self.kernel_size = kernel_size
        self.in_trivial  = in_fields[0]
        self.in_signflip = in_fields[1]
        self.in_regular  = in_fields[2]
        self.out_trivial  = out_fields[0]
        self.out_signflip = out_fields[1]
        self.out_regular  = out_fields[2]

        # register parameters for the reflection steerable kernels
        # due to the kernel symmetries, the number of parameters is reduced in comparison to unconstrained kernels
        # trivial->trivial (symmetric kernels)
        if self.out_trivial!=0 and self.in_trivial!=0:
            weights_triv2triv = torch.empty(self.out_trivial, self.in_trivial, kernel_size//2+1, kernel_size)
            self.register_parameter('weights_triv2triv', nn.Parameter(weights_triv2triv, requires_grad=True))
        # signflip->signflip (symmetric kernels)
        if self.out_signflip!=0 and self.in_signflip!=0:
            weights_sign2sign = torch.empty(self.out_signflip, self.in_signflip, kernel_size//2+1, kernel_size)
            self.register_parameter('weights_sign2sign', nn.Parameter(weights_sign2sign, requires_grad=True))
        # trivial->signflip (antisymmetric kernels)
        if self.out_signflip!=0 and self.in_trivial!=0:
            weights_triv2sign = torch.empty(self.out_signflip, self.in_trivial,  kernel_size//2, kernel_size)
            self.register_parameter('weights_triv2sign', nn.Parameter(weights_triv2sign, requires_grad=True))
        # signflip->trivial (antisymmetric kernels)
        if self.out_trivial!=0 and self.in_signflip!=0:
            weights_sign2triv = torch.empty(self.out_trivial, self.in_signflip, kernel_size//2, kernel_size)
            self.register_parameter('weights_sign2triv', nn.Parameter(weights_sign2triv, requires_grad=True))
        # trivial->regular (reflected copies of kernels)
        if self.out_regular!=0 and self.in_trivial!=0:
            weights_triv2reg = torch.empty(self.out_regular, self.in_trivial,  kernel_size, kernel_size)
            self.register_parameter('weights_triv2reg', nn.Parameter(weights_triv2reg, requires_grad=True))
        # signflip->regular (reflected, negated copies of kernels)
        if self.out_regular!=0 and self.in_signflip!=0:
            weights_sign2reg = torch.empty(self.out_regular, self.in_signflip, kernel_size, kernel_size)
            self.register_parameter('weights_sign2reg', nn.Parameter(weights_sign2reg, requires_grad=True))
        # regular->trivial (reflected copies of kernels)
        if self.out_trivial!=0 and self.in_regular!=0:
            weights_reg2triv = torch.empty(self.out_trivial, self.in_regular, kernel_size, kernel_size)
            self.register_parameter('weights_reg2triv', nn.Parameter(weights_reg2triv, requires_grad=True))
        # regular->signflip (reflected, negated copies of kernels)
        if self.out_signflip!=0 and self.in_regular!=0:
            weights_reg2sign = torch.empty(self.out_signflip, self.in_regular, kernel_size, kernel_size)
            self.register_parameter('weights_reg2sign', nn.Parameter(weights_reg2sign, requires_grad=True))
        # regular->regular (group convolution, reflected and permuted copies of kernels)
        if self.out_regular!=0 and self.in_regular!=0:
            weights_reg2reg = torch.empty(self.out_regular, 2*self.in_regular, kernel_size, kernel_size)
            self.register_parameter('weights_reg2reg', nn.Parameter(weights_reg2reg, requires_grad=True))

        # register parameters for the reflection steerable biases
        # due to the equivariance constraint, the number of parameters is reduced in comparison to unconstrained biases
        # scalar and regular fields have a 1-dim trivial invariant subspace per field while signflip fields have no
        # trivial invariant subspace
        if self.out_trivial!=0:
            bias_triv = torch.empty(self.out_trivial)
            self.register_parameter('bias_triv', nn.Parameter(bias_triv, requires_grad=True))
        if self.out_regular!=0:
            bias_reg  = torch.empty(self.out_regular)
            self.register_parameter('bias_reg', nn.Parameter(bias_reg, requires_grad=True))

        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights and biases
        
        Convolution weights are initialized according to He's weight initialization scheme
        Biases are set to zero
        """
        # use a factor of 2 instead of sqrt(2) in He's weight init since we are averaging fan_in and fan_out)
        std = 2/np.sqrt(self.kernel_size**2 * (
                        self.in_trivial +self.in_signflip +2*self.in_regular +
                        self.out_trivial+self.out_signflip+2*self.out_regular))
        # set the kernel parameters to random values sampled from a normal distribution
        if self.out_trivial!=0 and self.in_trivial!=0:
            self.weights_triv2triv.data = std * torch.randn_like(self.weights_triv2triv)
        if self.out_signflip!=0 and self.in_signflip!=0:
            self.weights_sign2sign.data = std * torch.randn_like(self.weights_sign2sign)
        if self.out_signflip!=0 and self.in_trivial!=0:
            self.weights_triv2sign.data = std * torch.randn_like(self.weights_triv2sign)
        if self.out_trivial!=0 and self.in_signflip!=0:
            self.weights_sign2triv.data = std * torch.randn_like(self.weights_sign2triv)
        if self.out_regular!=0 and self.in_trivial!=0:
            self.weights_triv2reg.data = std * torch.randn_like(self.weights_triv2reg)
        if self.out_regular!=0 and self.in_signflip!=0:
            self.weights_sign2reg.data = std * torch.randn_like(self.weights_sign2reg)
        if self.out_trivial!=0 and self.in_regular!=0:
            self.weights_reg2triv.data = std * torch.randn_like(self.weights_reg2triv)
        if self.out_signflip!=0 and self.in_regular!=0:
            self.weights_reg2sign.data = std * torch.randn_like(self.weights_reg2sign)
        if self.out_regular!=0 and self.in_regular!=0:
            self.weights_reg2reg.data = std * torch.randn_like(self.weights_reg2reg)
        # initialize biases to zero
        if self.out_trivial!=0:
            self.bias_triv.data = torch.zeros_like(self.bias_triv)
        if self.out_regular!=0:
            self.bias_reg.data = torch.zeros_like(self.bias_reg)

    def _reflect_kernel(self):
        """Manipulates kernel weights such that expanded kernels are spatially reflected (no action on channel dims)"""
        self.weights_triv2sign.data = -self.weights_triv2sign
        self.weights_sign2triv.data = -self.weights_sign2triv
        self.weights_triv2reg.data = self.weights_triv2reg.flip(2)
        self.weights_sign2reg.data = self.weights_sign2reg.flip(2)
        self.weights_reg2triv.data = self.weights_reg2triv.flip(2)
        self.weights_reg2sign.data = self.weights_reg2sign.flip(2)
        self.weights_reg2reg.data = self.weights_reg2reg.flip(2)

    def _expand_kernel(self, dtype=torch.float, device=torch.device('cpu')):
        """Kernel expansion method, building reflection steerable kernels from the kernel parameters

        The kernel expansion is performed in each forward pass, constructing the full reflection symmetric kernels from
        the (updated) kernel parameters.
        The full kernel consists of 9 blocks which map between all possible pairs of field types as visualized below
        (the spatial extent of the kernels is not shown):
                                   OUTPUT               BLOCKED KERNEL              INPUT
                                                 bound_in_1    bound_in_2
                                      _     ____________|____________|___________     _
                                     | |   |            |            |           |   | |
                      output trivial | |   | triv->triv | sign->triv | reg->triv |   | | input trivial
        bound_out_1__________________|_|   |____________|____________|___________|   |_|__________________bound_in_1
                                     | |   |            |            |           |   | |
                     output signflip | | = | triv->sign | sign->sign | reg->sign | * | | input signflip
        bound_out_2__________________|_|   |____________|____________|___________|   |_|__________________bound_in_2
                                     | |   |            |            |           |   | |
                      output regular | |   | triv->reg  | sign->reg  | reg->reg  |   | | input regular
                                     |_|   |____________|____________|___________|   |_|
        Each block comes with its own constraints and parameters and is therefore expanded individually.
        A visualization of the kernel symmetries in each block is given in Table 3 of the paper.

        Args:
            dtype (torch.dtype, optional): Datatype of the kernel. Defaults to torch.float
            device (torch.device, optional): Device on which the kernel is stored. Defaults to torch.device('cpu').

        Returns:
            torch.Tensor: expanded kernel of shape (c_out, c_in, self.kernel_size, self.kernel_size) where 
                c_in  = self.in_trivial  + self.in_signflip  + 2*self.in_regular and
                c_out = self.out_trivial + self.out_signflip + 2*self.out_regular
                are the full dimensionalities of the feature vectors, depending on the individual field multiplicities.

        """
        kernel = torch.zeros((self.out_trivial + self.out_signflip + 2*self.out_regular,
                              self.in_trivial  + self.in_signflip  + 2*self.in_regular,
                              self.kernel_size,
                              self.kernel_size),
                             dtype=dtype, device=device)

        # compute boundary indices which separate the blocks
        # column / input boundaries
        bound_in_1 = self.in_trivial
        bound_in_2 = self.in_trivial + self.in_signflip
        # row / output boundaries
        bound_out_1 = self.out_trivial
        bound_out_2 = self.out_trivial + self.out_signflip

        # fill the nine blocks of the kernel
        # trivial->trivial (symmetric kernels):
        #   fill upper half of kernels, including central row
        #   fill lower half of kernels, reflected (symmetric)
        if self.out_trivial!=0 and self.in_trivial!=0:
            kernel[:bound_out_1, :bound_in_1, :self.kernel_size//2+1] = self.weights_triv2triv
            kernel[:bound_out_1, :bound_in_1, self.kernel_size//2+1:] = \
                                                                self.weights_triv2triv[:,:,:self.kernel_size//2].flip(2)
        # signflip->signflip (symmetric kernels):
        #   fill upper half of kernels, including central row
        #   fill lower half of kernels, reflected (symmetric)
        if self.out_signflip!=0 and self.in_signflip!=0:
            kernel[bound_out_1:bound_out_2, bound_in_1:bound_in_2, :self.kernel_size//2+1] = self.weights_sign2sign
            kernel[bound_out_1:bound_out_2, bound_in_1:bound_in_2, self.kernel_size//2+1:] = \
                                                                self.weights_sign2sign[:,:,:self.kernel_size//2].flip(2)
        # trivial->signflip (antisymmetric kernels):
        #   1) fill upper half of kernels, excluding central row
        #   2) fill lower half of kernels, reflected and negated (antisymmetric)
        if self.out_signflip!=0 and self.in_trivial!=0:
                kernel[bound_out_1:bound_out_2, :bound_in_1, :self.kernel_size//2]   =  self.weights_triv2sign
                kernel[bound_out_1:bound_out_2, :bound_in_1, self.kernel_size//2+1:] = -self.weights_triv2sign.flip(2)
        # signflip->trivial (antisymmetric kernels):
        #   1) fill upper half of kernels, excluding central row
        #   2) fill lower half of kernels, reflected and negated (antisymmetric)
        if self.out_trivial!=0 and self.in_signflip!=0:
            kernel[:bound_out_1, bound_in_1:bound_in_2, :self.kernel_size//2]   =  self.weights_sign2triv
            kernel[:bound_out_1, bound_in_1:bound_in_2, self.kernel_size//2+1:] = -self.weights_sign2triv.flip(2)
        # trivial->regular (reflected copies of kernels):
        #   1) fill first channels (even indices) of regular rep feature fields
        #   2) fill second channels (odd indices) of regular rep feature fields, reflected
        if self.out_regular!=0 and self.in_trivial!=0:
            kernel[  bound_out_2::2, :bound_in_1] = self.weights_triv2reg
            kernel[1+bound_out_2::2, :bound_in_1] = self.weights_triv2reg.flip(2)
        # signflip->regular (reflected, negated copies of kernels):
        #   1) fill first channels (even indices) of regular rep feature fields
        #   2) fill second channels (odd indices) of regular rep feature fields, reflected, negated
        if self.out_regular!=0 and self.in_signflip!=0:
            kernel[  bound_out_2::2, bound_in_1:bound_in_2] =  self.weights_sign2reg
            kernel[1+bound_out_2::2, bound_in_1:bound_in_2] = -self.weights_sign2reg.flip(2)
        # regular->trivial (reflected copies of kernels):
        #   1) fill first channels (even indices) of regular rep feature fields
        #   2) fill second channels (odd indices) of regular rep feature fields, reflected
        if self.out_trivial!=0 and self.in_regular!=0:
            kernel[:bound_out_1,   bound_in_2::2] = self.weights_reg2triv
            kernel[:bound_out_1, 1+bound_in_2::2] = self.weights_reg2triv.flip(2)
        # regular->signflip (reflected, negated copies of kernels):
        #   1) fill first channels (even indices) of regular rep feature fields
        #   2) fill second channels (odd indices) of regular rep feature fields, reflected, negated
        if self.out_signflip!=0 and self.in_regular!=0:
            kernel[bound_out_1:bound_out_2,   bound_in_2::2] =  self.weights_reg2sign
            kernel[bound_out_1:bound_out_2, 1+bound_in_2::2] = -self.weights_reg2sign.flip(2)
        # regular->regular (group convolution, reflected and permuted copies of kernels):
        #   1) fill first channels (even indices) of regular rep output feature fields
        #   2) fill second channels (odd indices) of regular rep output feature fields, reflected, permuted:
        #       2.1) fill first channels (even indices) of regular rep input feature fields
        #       2.2) fill second channels (odd indices) of regular rep input feature fields
        if self.out_regular!=0 and self.in_regular!=0:
            kernel[  bound_out_2::2,   bound_in_2:  ] = self.weights_reg2reg
            kernel[1+bound_out_2::2,   bound_in_2::2] = self.weights_reg2reg[:,1::2].flip(2)
            kernel[1+bound_out_2::2, 1+bound_in_2::2] = self.weights_reg2reg[:, ::2].flip(2)

        return kernel

    def _expand_bias(self, dtype=torch.float, device=torch.device('cpu')):
        """Bias expansion method, building reflection steerable biases from the bias parameters

        The bias expansion is performed in each forward pass, constructing the full reflection invariant biases from
        the (updated) bias parameters.
        The biases for each field type come with their own constraint and parameters and are therefore expanded
        individually. Section 5.3.1 of the paper derives the constraints.

        Args:
            dtype (torch.dtype, optional): Datatype of the bias. Defaults to torch.float
            device (torch.device, optional): Device on which the bias is stored. Defaults to torch.device('cpu').

        Returns:
            torch.Tensor: expanded bias of shape (c_out) where 
                c_out = self.out_trivial + self.out_signflip + 2*self.out_regular
                is the full dimensionality of output feature vectors, depending on the individual field multiplicities.

        """
        bias = torch.zeros(self.out_trivial + self.out_signflip + 2*self.out_regular, dtype=dtype, device=device)
        if self.out_trivial!=0:
            bias[:self.out_trivial] = self.bias_triv
        if self.out_regular!=0:
            bias[  self.out_trivial+self.out_signflip::2] = self.bias_reg
            bias[1+self.out_trivial+self.out_signflip::2] = self.bias_reg
        return bias

    def _transport_pad(self, input):
        """Parallel transport padding operation on the Mobius strip (see Fig. 29 and Section 5.4.1 of the paper)

        Due to the twisted topology of the strip, the padded columns are spatially reflected.
        Transporters correspond to the Levi-Civita connection of the strip.
        The padding is performed for each field type individually since the parallel transport depends on the particular
        group representation.

        This implementation assumes an odd self.kernel_size such that the array is padded by a border of
        self.kernel_size//2 == (self.kernel_size-1)/2 pixels around its spatial dimensions. A convolution with "valid"
        boundary conditions (no additional padding) results then in an output tensor with the same spatial dimensions
        like the (non-padded) input tensor.

        Args:
            input (torch.Tensor): input feature field on the Mobius strip

        Returns:
            torch.Tensor: padded input feature field
        """
        N,C,W,L = input.shape
        size = self.kernel_size
        padded = torch.zeros((N, C, W+size-1, L+size-1), dtype=input.dtype, device=input.device)
        # compute boundary indices which separate different field types
        bound_1 = self.in_trivial
        bound_2 = self.in_trivial + self.in_signflip
        # keep interior region for all field types unchanged
        padded[:,:, size//2:-(size//2), size//2:-(size//2)] = input
        # scalar fields: cyclic padding with trivial action
        if self.in_trivial!=0:
            # right strip ===flip===> left padding region
            padded[:,:bound_1, size//2:-(size//2), :size//2] = input[:,:bound_1,:,-(size//2):].flip(2)
            # left strip ===flip===> right padding region
            padded[:,:bound_1, size//2:-(size//2), -(size//2):] = input[:,:bound_1,:, :size//2].flip(2)
        # signflip fields: cyclic padding with sign inversion
        if self.in_signflip!=0:
            # right strip ===flip+inversion===> left padding region
            padded[:,bound_1:bound_2, size//2:-(size//2), :size//2] = -input[:,bound_1:bound_2,:,-(size//2):].flip(2)
            # left strip ===flip+inversion===> right padding region
            padded[:,bound_1:bound_2, size//2:-(size//2), -(size//2):] = -input[:,bound_1:bound_2,:, :size//2].flip(2)
        # regular fields: cyclic padding with permutation action
        if self.in_regular!=0:
            # right strip ===flip+permutation===> left padding region
            # permutation: odd --> even regular field indices
            padded[:,  bound_2::2, size//2:-(size//2), :size//2] = input[:,1+bound_2::2,:,-(size//2):].flip(2)
            # permutation: even --> odd regular field indices
            padded[:,1+bound_2::2, size//2:-(size//2), :size//2] = input[:,  bound_2::2,:,-(size//2):].flip(2)
            # left strip ===flip+permutation===> right padding region
            # permutation: odd --> even regular field indices
            padded[:,  bound_2::2, size//2:-(size//2), -(size//2):] = input[:,1+bound_2::2,:, :size//2].flip(2)
            # permutation: even --> odd regular field indices
            padded[:,1+bound_2::2, size//2:-(size//2), -(size//2):] = input[:,  bound_2::2,:, :size//2].flip(2)
        return padded

    def forward(self, input):
        """Perform orientation independent convolution and bias summation on the Mobius strip"""
        padded = self._transport_pad(input)
        kernel = self._expand_kernel(input.dtype, input.device)
        bias = self._expand_bias(input.dtype, input.device)
        return F.conv2d(padded, kernel, bias=bias)


class EquivNonlin(nn.Module):
    """Reflection equivariant nonlinearity layer.

    Applies reflection equivariant nonlinearities as described in Section 5.3.2 of the paper.
    Each field type is assigned a different nonlinearity:
    - scalar fields apply ELU to their single invariant channel
    - signflip fields apply ELU to their absolute value after summing a bias. As this would result in a scalar field,
      the result is additionally multiplied with the sign of the feature. The bias is a learnable parameter.
    - regular feature fields apply ELUs to their two individual channels, such that the nonlinearity commutes with
      channel permutations (i.e. is reflection equivariant).
    Note that the output field types of the nonlinearities are chosen to coincide with their input field types.
    """
    def __init__(self, in_fields):
        """
        Args:
            in_fields (Tuple[int, int, int]): multiplicities of the input feature fields, transforming according to
                trivial, signflip and regular representations, respectively.

        """
        super(EquivNonlin, self).__init__()
        self.bound_1 = in_fields[0]
        self.bound_2 = in_fields[0] + in_fields[1]
        self.has_signflip = in_fields[1] > 0
        # allocate parameters for the signflip field nonlinearity
        if self.has_signflip:
            bias_sign = torch.zeros((1,in_fields[1],1,1))
            self.register_parameter('bias_sign', nn.Parameter(bias_sign, requires_grad=True))

    def forward(self, input):
        """Applies reflection equivariant nonlinearities to the input tensor"""
        output = torch.empty_like(input)
        # apply ELU nonlinearity to scalar fields and regular fields
        output[:, :self.bound_1] = F.elu(input[:, :self.bound_1]) # scalar
        output[:, self.bound_2:] = F.elu(input[:, self.bound_2:]) # regular
        # apply parameterized norm-nonlinearity to signflip fields
        if self.has_signflip:
            sign_fields = input[:, self.bound_1:self.bound_2]
            sign_fields = sign_fields/(sign_fields.abs() + 1e-8) * F.elu(sign_fields.abs() + self.bias_sign)
            output[:, self.bound_1:self.bound_2] = sign_fields
        return output


class MobiusPool(nn.Module):
    """Pooling operation with stride 2, reducing the spatial resolution of feature fields on the Mobius strip.

    Scalar and regular feature fields are pooled with a conventional max pooling operation, which is for these field
    types gauge independent. As the coefficients of signflip fields negate under gauge transformations, they are pooled
    based on their (gauge invariant) norm. A gauge independent alternative is average pooling.

    While the pooling operations are tested to be exactly gauge equivariant, the reduction of the spatial resolution
    interferes with their isometry equivariance. Specifically, the pooling operation is only isometry equivariant w.r.t.
    shifts by an even number of pixels.
    Note that the same issues apply for conventional Euclidean convolutions.

    """
    def __init__(self, in_fields):
        """
        Args:
            in_fields (Tuple[int, int, int]): multiplicities of the input feature fields, transforming according to
                trivial, signflip and regular representations, respectively.

        """
        super(MobiusPool, self).__init__()
        self.bound_1 = in_fields[0]
        self.bound_2 = in_fields[0] + in_fields[1]
        self.has_triv = in_fields[0] > 0
        self.has_sign = in_fields[1] > 0
        self.has_reg  = in_fields[2] > 0

    def _pool_triv(self, triv_fields, padding):
        return F.max_pool2d(triv_fields, kernel_size=2, stride=2, padding=padding)

    def _pool_sign(self, sign_fields, padding, avg_pool=False):
        if avg_pool:
            # average pooling is automatically gauge equivariant for sign-flip fields,
            # however, it performs worse than the norm-based max pooling
            return F.avg_pool2d(sign_fields, kernel_size=2, stride=2, padding=padding)
        else:
            # pool signflip fields based on their maximal norm, preserve sign for equivariance
            # (would otherwise map to scalar fields)
            # max pooling of positive and negative tensor, gives max and negated min entries
            pooled_sign_pos = F.max_pool2d( sign_fields, kernel_size=2, stride=2, padding=padding)
            pooled_sign_neg = F.max_pool2d(-sign_fields, kernel_size=2, stride=2, padding=padding)
            # get boolean mask signaling whether the value with maximal norm is positive or negative
            pos_geq_neg = pooled_sign_pos >= pooled_sign_neg
            # fill in values with maximal norm
            pooled_sign = torch.empty_like(pooled_sign_pos)
            pooled_sign[ pos_geq_neg] = pooled_sign_pos[ pos_geq_neg] # positives are kept positive
            pooled_sign[~pos_geq_neg] = -pooled_sign_neg[~pos_geq_neg] # recover negative entries from absolute values
            return pooled_sign

    def _pool_reg(self, reg_fields, padding):
        return F.max_pool2d(reg_fields, kernel_size=2, stride=2, padding=padding)

    def forward(self, input):
        """Apply the pooling operation with a stride of 2 pixels"""
        # if the length L of the strip is odd, we pad one reflected + gauge transformed column of
        # pixels from the otherside of the cut
        N,C,W,L = input.shape
        if L%2 == 1:
            padded = torch.empty((N,C,W,L+1), dtype=input.dtype, device=input.device)
            padded[...,:L] = input
            if self.has_triv:
                padded[:,:self.bound_1,:,-1] = input[:,:self.bound_1,:,0].flip(2)
            if self.has_sign:
                padded[:,self.bound_1:self.bound_2,:,-1] = -input[:,self.bound_1:self.bound_2,:,0].flip(2)
            if self.has_reg:
                padded[:,self.bound_2::2,:,-1] = input[:,self.bound_2+1::2,:,0].flip(2)
                padded[:,self.bound_2+1::2,:,-1] = input[:,self.bound_2::2,:,0].flip(2)
            input = padded
        # apply pooling ops for the individual field types
        # if the strip width W is odd, symmetrize pooling region by zero-padding on both sides, then average results
        # this is done via .flip(2) -- however, this is NOT a gauge transformation (and thus there is no action of rho)
        pooled = []
        if self.has_triv:
            triv_fields = input[:,:self.bound_1]
            if W%2 == 0:
                pooled_triv = self._pool_triv(triv_fields, padding=0)
            else:
                pooled_triv = self._pool_triv(triv_fields, padding=(1,0))
                pooled_triv += self._pool_triv(triv_fields.flip(2), padding=(1,0)).flip(2)
                pooled_triv /= 2.
            pooled.append(pooled_triv)
        if self.has_sign:
            sign_fields = input[:, self.bound_1:self.bound_2]
            if W%2 == 0:
                pooled_sign = self._pool_sign(sign_fields, padding=0)
            else:
                pooled_sign = self._pool_sign(sign_fields, padding=(1,0))
                pooled_sign += self._pool_sign(sign_fields.flip(2), padding=(1,0)).flip(2)
                pooled_sign /= 2.
            pooled.append(pooled_sign)
        if self.has_reg:
            reg_fields = input[:, self.bound_2:]
            if W%2 == 0:
                pooled_reg = self._pool_reg(reg_fields, padding=0)
            else:
                pooled_reg = self._pool_reg(reg_fields, padding=(1,0))
                pooled_reg += self._pool_reg(reg_fields.flip(2), padding=(1,0)).flip(2)
                pooled_reg /= 2.
            pooled.append(pooled_reg)
        pooled = torch.cat(pooled, dim=1)
        return pooled


class MobiusPadNaive(nn.Module):
    """ Naive padding, transporting features according to the trivial connection.

    This padding operation respects the strip's topology by flipping the padded stripes spatially.
    However, it does not apply a (non-trivial) gauge transformation as done in MobiusConv._transport_pad.
    This operation is used in the conventional CNN baseline, which is agnostic of field types (gauge transformation
    laws). It can not be used in a reflection equivariant model.
    """
    def __init__(self, kernel_size):
        """
        Args:
            kernel_size (int): Size of the convolution kernel, which implies the number of pixels to be padded.
                Assumed to be odd.

        """
        super(MobiusPadNaive, self).__init__()
        assert kernel_size%2 == 1
        self.kernel_size = kernel_size

    def forward(self, input):
        N,C,W,L = input.shape
        size = self.kernel_size
        padded = torch.zeros((N, C, W+size-1, L+size-1), dtype=input.dtype, device=input.device)
        # keep interior region unchanged
        padded[:,:, size//2:-(size//2), size//2:-(size//2)] = input
        # cyclic padding *without* gauge action
        padded[:,:, size//2:-(size//2), :size//2] = input[:,:,:,-(size//2):].flip(2)
        padded[:,:, size//2:-(size//2), -(size//2):] = input[:,:,:, :size//2].flip(2)
        return padded
