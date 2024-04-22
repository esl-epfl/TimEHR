"""

"""

# stdlib
from typing import List

# third party
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(
        self,
        channels_img: int,
        image_size: int,
        features_d: int,
        d_conditional: int = 9,
        conditional: bool = False,
        kernel_size: int = 4,
    ) -> None:
        super(Critic, self).__init__()

        self.channels_img = channels_img
        self.image_size = image_size
        self.features_d = features_d
        self.d_conditional = d_conditional
        self.conditional = conditional
        self.kernel_size = kernel_size

        self.stride = 2
        self.padding = (self.kernel_size - 2) // 2

        if conditional:
            self.embed = nn.Linear(
                d_conditional, self.image_size * self.image_size, bias=False
            )
            self.d_ch_cond = 1
            pass
        else:
            self.d_ch_cond = 0
            pass

        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img + self.d_ch_cond,
                features_d,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, self.kernel_size, self.stride, self.padding)
            self._block(
                features_d, features_d * 2, self.kernel_size, self.stride, self.padding
            ),
            self._block(
                features_d * 2,
                features_d * 4,
                self.kernel_size,
                self.stride,
                self.padding,
            ),
            self._block(
                features_d * 4,
                features_d * 8,
                self.kernel_size,
                self.stride,
                self.padding,
            ),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if y is not None:
            y = self.embed(y).view(-1, 1, self.image_size, self.image_size)
            x = torch.cat([x, y], dim=1)
        return self.disc(x)


class Generator1(nn.Module):
    def __init__(
        self,
        channels_noise: int,
        channels_img: int,
        image_size: int,
        features_g: int,
        d_conditional: int = 9,
        conditional: bool = False,
        kernel_size: int = 4,
    ) -> None:
        super(Generator1, self).__init__()

        self.channels_noise = channels_noise
        self.channels_img = channels_img
        self.image_size = image_size
        self.features_g = features_g
        self.d_conditional = d_conditional
        self.conditional = conditional
        self.kernel_size = kernel_size

        self.stride = 2
        self.padding = (self.kernel_size - 2) // 2

        if conditional:
            self.embed = nn.Linear(d_conditional, 50, bias=False)
            self.d_ch_cond = 50
            pass
        else:
            self.d_ch_cond = 0
            pass

        if image_size == 64:
            self.net = nn.Sequential(
                # Input: N x channels_noise x 1 x 1
                self._block(
                    channels_noise + self.d_ch_cond, features_g * 16, 4, 1, 0
                ),  # img: 4x4
                self._block(
                    features_g * 16,
                    features_g * 8,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                ),  # img: 8x8
                self._block(
                    features_g * 8,
                    features_g * 4,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                ),  # img: 16x16
                self._block(
                    features_g * 4,
                    features_g * 2,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                ),  # img: 32x32
                # self._block(features_g * 2, features_g, 4, 2, 1),  # img: 64x64
                nn.ConvTranspose2d(
                    features_g * 2,
                    channels_img,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                ),
                # Output: N x channels_img x 128 x 128
                nn.Tanh(),
                # nn.Sigmoid(),
            )
        elif image_size == 128:
            self.net = nn.Sequential(
                # Input: N x channels_noise x 1 x 1
                self._block(
                    channels_noise + self.d_ch_cond, features_g * 16, 4, 1, 0
                ),  # img: 4x4
                self._block(
                    features_g * 16,
                    features_g * 8,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                ),  # img: 8x8
                self._block(
                    features_g * 8,
                    features_g * 4,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                ),  # img: 16x16
                self._block(
                    features_g * 4,
                    features_g * 2,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                ),  # img: 32x32
                self._block(features_g * 2, features_g, 4, 2, 1),  # img: 64x64
                nn.ConvTranspose2d(
                    features_g,
                    channels_img,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                ),
                # Output: N x channels_img x 128 x 128
                nn.Tanh(),
                # nn.Sigmoid(),
            )

    def _block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        # x shape is N x channels_noise x 1 x 1
        if y is not None:
            y = self.embed(y).unsqueeze(-1).unsqueeze(-1)  # N x 50 x 1 x 1
            x = torch.cat([x, y], dim=1)
        x = self.net(x)

        # x[:,0,:,:][x[:,1,:,:]<0]=0
        # x[:,0,:,:] =  x[:,0,:,:].masked_fill(x[:,1,:,:]<0,0)

        return x


def initialize_weights(model: nn.Module) -> None:
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class Block(nn.Module):  # this is the for generator pix
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        down: bool = True,
        act: str = "relu",
        use_dropout: bool = False,
    ) -> None:
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            (
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    4,
                    2,
                    1,
                    bias=False,
                    padding_mode="reflect",
                )
                if down
                else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels, affine=True),
            # nn.LayerNorm((out_channels*4,in_channels,in_channels)),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Gen_pix(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        features: int = 64,
        d_conditional: int = 9,
        conditional: bool = False,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()

        self.channels_img = in_channels
        self.image_size = image_size
        self.features = features
        self.d_conditional = d_conditional
        self.conditional = conditional
        self.kernel_size = kernel_size

        self.stride = 2
        self.padding = (self.kernel_size - 2) // 2

        if conditional:
            self.embed = nn.Linear(
                d_conditional, self.image_size * self.image_size, bias=False
            )
            self.d_ch_cond = 1
            pass
        else:
            self.d_ch_cond = 0
            pass

        self.initial_down = nn.Sequential(
            nn.Conv2d(
                in_channels + self.d_ch_cond, features, 4, 2, 1, padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
        )
        self.down1 = Block(
            features, features * 2, down=True, act="leaky", use_dropout=False
        )
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(
            features * 8, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(
            features * 2 * 2, features, down=False, act="relu", use_dropout=False
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, in_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

        self.downsampling = nn.Sequential()
        self.downsampling.add_module("initial_down", self.initial_down)

        if image_size == 16:
            self.downsampling.add_module("down1", self.down1)

    def forward(self, x: torch.Tensor, sta: torch.Tensor = None) -> torch.Tensor:
        # x = torch.cat([x, y_partial], dim=1)
        if sta is not None:
            # shape = (batch_size, 1, 256, 256)
            sta = self.embed(sta).view(-1, 1, self.image_size, self.image_size)
            x = torch.cat([x, sta], dim=1)

        d1 = self.initial_down(x)  # shape = (batch_size, f, /2, /2)

        d2 = self.down1(d1)  # shape = (batch_size, 2f, /4, /4)
        d3 = self.down2(d2)  # shape = (batch_size, 4f, /8, /8)
        d4 = self.down3(d3)  # shape = (batch_size, 8f, /16, /16)
        d5 = self.down4(d4)  # shape = (batch_size, 8f, /32, /32)
        # d6 = self.down5(d5) # shape = (batch_size, 8f, /64, /64)
        # d7 = self.down6(d6) # shape = (batch_size, 8f, /128, /128)

        bottleneck = self.bottleneck(d5)  # shape = (batch_size, 8f, /256=1, /256=1)

        up1 = self.up1(bottleneck)  # shape = (batch_size, 8f, *2, *2)
        # up2 = self.up2(torch.cat([up1, d7], 1)) # shape = (batch_size, 8f, *4, *4)
        # up3 = self.up3(torch.cat([up2, d6], 1)) # shape = (batch_size, 8f, *8, *8)

        up4 = self.up4(torch.cat([up1, d5], 1))  # shape = (batch_size, 8f, *16, *16)

        up5 = self.up5(torch.cat([up4, d4], 1))  # shape = (batch_size, 4f, *32, *32)

        up6 = self.up6(torch.cat([up5, d3], 1))  # shape = (batch_size, 2f, *64, *64)

        up7 = self.up7(torch.cat([up6, d2], 1))  # shape = (batch_size, f, *128, *128)

        return self.final_up(
            torch.cat([up7, d1], 1)
        )  # shape = (batch_size, n_ch, *256, *256)


class CNNBlock(nn.Module):  # this is the discriminator pix
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                4,
                stride,
                1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels, affine=True),
            # nn.LayerNorm((1,64,64)),
            nn.LeakyReLU(0.2),
            # nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Disc_pix(nn.Module):
    def __init__(
        self,
        in_channels: int,
        image_size: int,
        features: List = [64, 128, 256, 512],
        d_conditional: int = 9,
        conditional: bool = False,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()

        self.channels_img = in_channels
        self.image_size = image_size
        self.features = features
        self.d_conditional = d_conditional
        self.conditional = conditional
        self.kernel_size = kernel_size

        self.stride = 2
        self.padding = (self.kernel_size - 2) // 2

        if conditional:
            self.embed = nn.Linear(
                d_conditional, self.image_size * self.image_size, bias=False
            )
            # self.embed2 = nn.Linear(d_conditional, 1, bias=False)
            self.d_ch_cond = 1
            pass
        else:
            self.d_ch_cond = 0
            pass

        # d_additional = 1 if conditional else 0
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels + self.d_ch_cond,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(
                    in_channels, feature, stride=1 if feature == features[-1] else 2
                ),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, sta: torch.Tensor = None
    ) -> torch.Tensor:
        # x.shape = (batch_size, channels, height, width)
        # y.shape = (batch_size, channels, height, width)
        # shape = (batch_size, channels*2, height, width)
        x = torch.cat([x, y], dim=1)

        if sta is not None:
            sta = self.embed(sta).view(-1, 1, self.image_size, self.image_size)

            # expand to height, width
            sta = sta.expand(-1, -1, self.image_size, self.image_size)

            # sta =sta.view(-1, 1, self.image_size, self.image_size)
            x = torch.cat([x, sta], dim=1)
        # shape = (batch_size, features[0], height/2, width/2)
        x = self.initial(x)
        x = self.model(x)  # shape = (batch_size, 1, 30, 30)
        return x


def test_pix() -> None:
    img_size = 128
    N, in_channels, H, W = 8, 2, img_size, img_size
    # noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    print("x.shape", x.shape)

    gen = Gen_pix(1, H, 8, conditional=True)
    disc = Disc_pix(2, H, conditional=False, features=[8, 16, 32])
    # print number of params
    print(sum(p.numel() for p in gen.parameters() if p.requires_grad))

    print(gen(x).shape)
    print(disc(x[:, [0]], x[:, [0]]).shape)
    # assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


def test_cwgan() -> None:

    img_size = 128
    N, in_channels, H, W = 8, 2, img_size, img_size
    # noise_dim = 100
    x = torch.randn((N, in_channels, H, W))

    gen = Generator1(100, in_channels, H, 8, conditional=False)
    noise = torch.randn((N, 100, 1, 1))
    fake = gen(noise)
    print("x.shape", x.shape)
    print("fake.shape", fake.shape)

    print(
        "number of params in gen",
        sum(p.numel() for p in gen.parameters() if p.requires_grad),
    )


if __name__ == "__main__":
    # test_AE()
    test_cwgan()
    test_pix()
# test()
