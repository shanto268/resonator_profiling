import numpy as np

import abc
import svgwrite


def get_unit_prefix(value):
    value = abs(value)
    if value >= 10**30:
        return "Q"
    elif value >= 10**27:
        return "R"
    elif value >= 10**24:
        return "Y"
    elif value >= 10**21:
        return "Z"
    elif value >= 10**18:
        return "E"
    elif value >= 10**15:
        return "P"
    elif value >= 10**12:
        return "T"
    elif value >= 10**9:
        return "G"
    elif value >= 10**6:
        return "M"
    elif value >= 10**3:
        return "k"
    elif value >= 1:
        return ""
    elif value < 1e-27:
        return "q"
    elif value < 1e-24:
        return "r"
    elif value < 1e-21:
        return "y"
    elif value < 1e-18:
        return "z"
    elif value < 1e-15:
        return "a"
    elif value < 1e-12:
        return "f"
    elif value < 1e-9:
        return "p"
    elif value < 1e-6:
        return "n"
    elif value < 1e-3:
        return "u"
    elif value < 1:
        return "m"

prefix_bases = {
    "Q": 10**30,
    "R": 10**27,
    "Y": 10**24,
    "Z": 10**21,
    "E": 10**18,
    "P": 10**15,
    "T": 10**12,
    "G": 10**9,
    "M": 10**6,
    "k": 10**3,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
    "a": 1e-18,
    "z": 1e-21,
    "y": 1e-24,
    "r": 1e-27,
    "q": 1e-30,
}

def round_to_leading_digits(number, leading_digits):
    if number == 0:
        return 0
    # Determine the number of digits in the integer part of the number
    digits = int(np.floor(np.log10(abs(number))) + 1)
    # Calculate the factor to scale the number
    factor = 10 ** (digits - leading_digits)
    # Round the scaled number
    rounded_number = round(number / factor) * factor
    return rounded_number

def format_value(value, precision):
    prefix = get_unit_prefix(value)
    if prefix != "":
        value /= prefix_bases[prefix]
    rounded_val = round_to_leading_digits(value, precision)
    num_of_digits = precision - int(np.floor(np.log10(abs(value))) + 1)
    if num_of_digits >= 1:
        string_value = f"{rounded_val:.{num_of_digits}f}"
    else:
        string_value = int(rounded_val)
    return f"{string_value}{prefix}"



class ABCDMatrix(abc.ABC):
    _plot_labels = True # If True, plot will contain the length, impedance, capacitance, etc. labels of all elements
    _labels_precision = 3 # Number of decimal places to show in the labels

    @abc.abstractmethod
    def abcd(self, w):
        ...

    @abc.abstractmethod
    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        ...

    def __mul__(self, other: 'ABCDMatrix'):
        return ABCDProduct(self, other)

    def sparams(self, w, z0):
        (a, b), (c, d) = self.abcd(w)
        b /= z0
        c *= z0
        denom = a + b + c + d
        return [
            (a + b - c - d)/denom,
            2*(a*d - b*c)/denom,
            2/denom,
            (-a + b - c + d)/denom,
        ]

    def yparams(self, w):
        (a, b), (c, d) = self.abcd(w)
        return [
            d/b,
            (b*c-a*d)/b,
            -1/b,
            a/b,
        ]

    def zparams(self, w):
        (a, b), (c, d) = self.abcd(w)
        return [
            a/c,
            (a*d-b*c)/c,
            1/c,
            d/c,
        ]

    def input_impedance(self, w, z0):
        [z11, z12, z21, z22] = self.zparams(w)
        return z11 - z12*z21 / (z22 + z0)

    def i(self, w):
        return 1 if np.isscalar(w) else np.ones_like(w, dtype=complex)

    def _repr_svg_(self, format_params=None):
        fmt = dict(
            cpw_len_func=(lambda l: 25),
            # cpw_len_func=(lambda l: l/1e-6),
        )
        if format_params is not None:
            fmt.update(format_params)

        # dwg = svgwrite.Drawing(size=('50%','50%'))
        dwg = svgwrite.Drawing(size=('50%', '500px'))  # 50% width, max 300px height

        gs, bbox, port_out = self._svg_group(
                                dwg,
                                fmt,
                                plot_labels = self._plot_labels,
                                labels_precision = self._labels_precision,
                            )
        (xmin, ymin), (xmax, ymax) = bbox
        dwg.viewbox(xmin-5, ymin-5, xmax-xmin+10, ymax-ymin+10)
        return dwg.tostring()

    def _draw_gnd(self, dwg, parent, fmt):
        gnd = parent.add(dwg.g())
        gnd.add(dwg.line((0, 0), (0, 10), stroke='#000'))
        gnd.add(dwg.line((-6, 10), (6, 10), stroke='#000'))
        gnd.add(dwg.line((-4, 13), (4, 13), stroke='#000'))
        gnd.add(dwg.line((-2, 16), (2, 16), stroke='#000'))
        return gnd

    @staticmethod
    def _draw_translate_last(el, tx, ty=None):
        new_transform = f'translate({svgwrite.utils.strlist([tx, ty])})'
        old_transform = el.attribs.get(el.transformname, '')
        el[el.transformname] = f'{new_transform} {old_transform}'.strip()

    @staticmethod
    def _draw_rotate_last(el, angle, center=None):
        new_transform = f'rotate({svgwrite.utils.strlist([angle, center])})'
        old_transform = el.attribs.get(el.transformname, '')
        el[el.transformname] = f'{new_transform} {old_transform}'.strip()

class ABCDProduct(ABCDMatrix):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def abcd(self, w):
        l = self.left.abcd(w)
        r = self.right.abcd(w)
        return np.array([[
            l[0,0]*r[0,0] + l[0,1]*r[1,0],
            l[0,0]*r[0,1] + l[0,1]*r[1,1],
        ],[
            l[1,0]*r[0,0] + l[1,1]*r[1,0],
            l[1,0]*r[0,1] + l[1,1]*r[1,1],
        ]])

    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        gsl, bbl, pol = self.left._svg_group(dwg, fmt, plot_labels=plot_labels, labels_precision=labels_precision)
        gsr, bbr, por = self.right._svg_group(dwg, fmt, plot_labels=plot_labels, labels_precision=labels_precision)
        for g in gsr:
            self._draw_translate_last(g, *pol)
        bbox = np.array([bbl, np.array(bbr) + np.array(pol)])
        bbox = [np.min(bbox, axis=0)[0], np.max(bbox, axis=0)[1]]
        return gsl + gsr, bbox, (pol[0] + por[0], pol[1] + por[1])

class ABCDInverse(ABCDMatrix):
    def __init__(self, m):
        self.m = m
    def abcd(self, w):
        ((a,b),(c,d)) = self.m.abcd(w)
        det = a*d - b*c
        return np.array([[d, b],[c, a]])/det
    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        gs, bbox, po = self.m._svg_group(dwg, fmt, plot_labels, labels_precision)
        for g in gs:
            self._draw_translate_last(g, -po[0], -po[1])
            self._draw_rotate_last(g, 180, (0,0))
        bbox = -np.array(bbox)[::-1] + po
        return gs, bbox, po

class ABCDIdentity(ABCDMatrix):
    def abcd(self, w):
        i = self.i(w)
        return np.array([[i, 0*i],[0*i, i]])
    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        return [], ((0,0), (0,0)), (0, 0)

class ABCDSeriesImpedance(ABCDMatrix):
    def __init__(self, z, label: str = ""):
        self.z = z
        self.label = label

    def abcd(self, w):
        i = self.i(w)
        return np.array([[i, self.z*i], [0*i , i]])

    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        g = dwg.add(dwg.g())
        g.add(dwg.line((0,0), (10,0), stroke='#000'))
        g.add(dwg.text('Z', (16, 5)))
        g.add(dwg.rect((10,-10), (20,20), stroke='#000', fill='none'))
        g.add(dwg.line((30,0), (40,0), stroke='#000'))
        if plot_labels:
            if self.label == "":
                self.label = format_value(self.z, labels_precision) + "Ω"
            text_obj = dwg.text(self.label, (33, -1), style = "font-size:5px")
            g.add(text_obj)
        return [g], ((0,-10),(40,10)), (40, 0)


class ABCDParallelAdmittance(ABCDMatrix):
    def __init__(self, y, label: str = ""):
        self.y = y
        self.label = label

    def abcd(self, w):
        i = self.i(w)
        return np.array([[i, 0*i], [self.y*i, i]])

    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        g = dwg.add(dwg.g())
        g.add(dwg.line((0,0), (20,0), stroke='#000'))
        g.add(dwg.line((10,0), (10,10), stroke='#000'))
        g.add(dwg.text('Y', (5.4, 25)))
        g.add(dwg.rect((0,10), (20,20), stroke='#000', fill='none'))
        self._draw_gnd(dwg, g, fmt).translate(10,30)
        if plot_labels:
            if self.label == "":
                self.label = format_value(self.y, labels_precision) + "S"
            text_obj = dwg.text(self.label, (12, 37), style = "font-size:5px")
            g.add(text_obj)
        return [g], ((0,0),(20,50)), (20, 0)


class ABCDTEMTransmissionLine(ABCDMatrix):
    """_summary_

    Parameters
    ----------
    l : float
        Length in um
    z0 : float
        Characteristic impedance in Ohm
    vp : float
        Phase velocity
    alpha : float, optional
        ? Defaults to 0.
    """
    def __init__(self, l: float, z0: float, vp: float, alpha: float = 0, label: str = ""):
        self.l = l
        self.z0 = max(z0, 1e-10)
        self.vp = vp
        self.alpha = alpha
        self.label = label

    def abcd(self, w):
        bl = self.l*(w/self.vp - 1j*self.alpha)
        return np.array([[np.cos(bl), 1j*np.sin(bl)*self.z0],
                         [1j*np.sin(bl)/self.z0, np.cos(bl)]])

    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        l = fmt['cpw_len_func'](self.l)
        g = dwg.add(dwg.g())
        g.add(dwg.line((0,0), (10,0), stroke='#000'))
        g.add(dwg.ellipse((10,0), (2.5,10), stroke='#000', fill='none'))
        g.add(dwg.line((10,-10), (10+l,-10), stroke='#000'))
        g.add(dwg.line((10, 10), (10+l, 10), stroke='#000'))
        g.add(dwg.path([f'M{10+l},-10', f'A2.5,10 1 0,1 {10+l},10'], stroke='#000', fill='none'))
        g.add(dwg.line((12.5+l,0), (22.5+l,0), stroke='#000'))
        if plot_labels:
            if self.label == "":
                self.label = format_value(self.l, labels_precision) + "m"
            g.add(dwg.text(self.label, (14, 1), style = "font-size:5px"))
        # return [g], ((0,-10),(22.5+l,10)), (22.5+l, 0)
        return [g], ((0,-15),(22.5+l,10)), (22.5+l, 0)


class ABCDSeriesCapacitance(ABCDMatrix):
    def __init__(self, c, label: str = ""):
        self.c = c
        self.label = label

    def abcd(self, w):
        i = self.i(w)
        return np.array([[i, -1j/(w*self.c)], [0*i, i]])

    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        l = 10
        g = dwg.add(dwg.g())
        g.add(dwg.line((0,0), (10,0), stroke='#000'))
        g.add(dwg.line((10,-10), (10,10), stroke='#000'))
        g.add(dwg.line((13,-10), (13,10), stroke='#000'))
        g.add(dwg.line((13,0), (13+l,0), stroke='#000'))
        if plot_labels:
            if self.label == "":
                self.label = format_value(self.c, labels_precision) + "F"
            g.add(dwg.text(self.label, (5, -12), style = "font-size:5px"))
        return [g], ((0,-14),(13+l,10)), (13+l, 0)


class ABCDParallelCapacitance(ABCDMatrix):
    def __init__(self, c, label: str = ""):
        self.c = c
        self.label = label

    def abcd(self, w):
        i = self.i(w)
        return np.array([[i, 0*i], [1j*w*self.c, i]])
    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        g = dwg.add(dwg.g())
        g.add(dwg.line((0,0), (20,0), stroke='#000'))
        g.add(dwg.line((10,0), (10,10), stroke='#000'))
        g.add(dwg.line((0,10), (20,10), stroke='#000'))
        g.add(dwg.line((0,13), (20,13), stroke='#000'))

        if plot_labels:
            if self.label == "":
                self.label = format_value(self.c, labels_precision) + "F"
            g.add(dwg.text(self.label, (11, 20), style = "font-size:5px"))

        self._draw_gnd(dwg, g, fmt).translate(10,13)
        return [g], ((0,0),(24,33)), (20, 0)

class ABCDSeriesInductance(ABCDMatrix):
    def __init__(self, l, label: str = ""):
        self.l = l
        self.label = label

    def abcd(self, w):
        i = self.i(w)
        return np.array([[i, 1j*w*self.l], [0*i,i]])
    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        r1 = 10/3
        r2 = 10/9
        g = dwg.add(dwg.g())
        g.add(dwg.line((0,0), (10,0), stroke='#000'))
        p = 10 + 2*r1
        commands = ['M10,0', f'A{r1},{1.2*r1} 1 0,1 {10+2*r1},0']
        for _ in range(3):
            commands += [
                f'A{r2},{r2} 1 0,1 {p-2*r2},0'
                f'A{r1},{1.2*r1} 1 0,1 {p-2*r2+2*r1},0'
            ]
            p = p-2*r2+2*r1
        g.add(dwg.path(commands, stroke='#000', fill='none'))
        g.add(dwg.line((30,0), (40,0), stroke='#000'))

        if plot_labels:
            if self.label == "":
                self.label = format_value(self.l, labels_precision) + "H"
            g.add(dwg.text(self.label, (12, -6), style = "font-size:5px"))

        return [g], ((0,-10),(40,10)), (40, 0)

class ABCDParallelInductance(ABCDMatrix):
    def __init__(self, l, label: str = ""):
        self.l = l
        self.label = label

    def abcd(self, w):
        i = self.i(w)
        return np.array([[i, 0*i], [-1j/(w*self.l), i]])
    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        r1 = 10/3
        r2 = 10/9
        g = dwg.add(dwg.g())
        g.add(dwg.line((0,0), (20,0), stroke='#000'))
        g.add(dwg.line((10,0), (10,10), stroke='#000'))
        p = 10 + 2*r1
        commands = ['M10,10', f'A{1.2*r1},{r1} 1 0,1 10,{10+2*r1}']
        for _ in range(3):
            commands += [
                f'A{r2},{r2} 1 0,1 10,{p-2*r2}'
                f'A{1.2*r1},{r1} 1 0,1 10,{p-2*r2+2*r1}'
            ]
            p = p-2*r2+2*r1
        g.add(dwg.path(commands, stroke='#000', fill='none'))

        if plot_labels:
            if self.label == "":
                self.label = format_value(self.l, labels_precision) + "H"
            text = dwg.text(self.label, (11, 37), style = "font-size:5px")
            # self._draw_rotate_last(text, 180, (10,37))
            # g.add(text)
            g.add(dwg.text(self.label, (11, 37), style = "font-size:5px"))
            

        self._draw_gnd(dwg, g, fmt).translate(10,30)
        return [g], ((0,0),(25,50)), (20, 0)

class ABCDCPWWithABs(ABCDMatrix):
    """
    ab_pattern: A list of tuples of two values, indicating the distances between airbridges (ABs) on this line.
                Each tuple corresponds to a distance between two ABs or an AB and the end of the waveguide.
                The first element of the tuple indicates the absolute distance, and the second indicates the weight
                for the leftover distance to be placed in this section.
                In total, `len(ab_pattern) - 1` airbridges are added.

                For example, to place an AB at the middle of the line, use `ab_pattern = [(0,1), (0,1)]`.
                To place one AB at 100 units from the start of the line and a second one at 200 units from
                the end, use `ab_pattern = [(100,0), (0,1), (200,0)]`.
    """
    def __init__(self, ltot, z0, vp, ab_pattern=None, c_ab=0.0, alpha=0):
        if ab_pattern is None or len(ab_pattern) <= 1:
            ab_pattern = [(0,1)]
        ab_pattern = np.array(ab_pattern)
        sum_abs, sum_rel = ab_pattern.sum(axis=0)
        if sum_rel == 0:
            ab_pattern[:,1] += 1
            sum_abs, sum_rel = ab_pattern.sum(axis=0)
        lens = (ab_pattern*[1, (ltot - sum_abs)/sum_rel]).sum(axis=1)
        self.value = ABCDTEMTransmissionLine(lens[0], z0, vp, alpha)
        # print(lens[0]/1e-6)
        for l in lens[1:]:
            self.value = self.value * \
                         ABCDParallelCapacitance(c_ab) * \
                         ABCDTEMTransmissionLine(l, z0, vp, alpha)
            # print(l/1e-6)

    def abcd(self, w):
        return self.value.abcd(w)
    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        return self.value._svg_group(dwg, fmt, plot_labels, labels_precision)

class ABCDTJunction(ABCDMatrix):
    """
    A shorted ABCD matrix is connected to the extra branch of the T-junction

     in
      |
      v
    __|__   __________________   _________
    |   |   |                |   |       |
    | T |->-| branching_abcd |->-| short |
    |___|   |________________|   |_______|
      |
      v
      |
     out
    """
    def __init__(self, branching_abcd):
        self.branching_abcd = branching_abcd

    def abcd(self, w):
        ((a,b),(c,d)) = self.branching_abcd.abcd(w)
        i = self.i(w)
        return np.array([[i,  0*i],
                         [d/b , i]])
    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        gs, bbox, po = self.branching_abcd._svg_group(dwg, fmt, plot_labels, labels_precision)
        for g in gs:
            self._draw_rotate_last(g, 90, (0,0))
            self._draw_translate_last(g, 10, 0)
        g = dwg.add(dwg.g())
        g.add(dwg.line((-5,0), (30,0), stroke='#000'))
        gnd = self._draw_gnd(dwg, g, fmt)
        gnd.translate(10-po[1], po[0])
        gs.append(g)
        (xmin, ymin), (xmax, ymax) = bbox
        bbox = ((-ymax+10 - 5, xmin), (-ymin+10+5, max(xmax, po[0]+20)))
        return gs, bbox, (30, 0)


class ABCDParallelNetwork(ABCDMatrix):
    def __init__(self, up, down):
        self.up = up
        self.down = down

    def abcd(self, w):
        # (y11, y12), (y21, y22) = self.up.admittance(w) + self.down.admittance(w)
        [y11, y12, y21, y22] = np.array(self.up.yparams(w)) + np.array(self.down.yparams(w))
        return np.array([[-y22, -self.i(w)],[y21*y12 - y11*y22, -y11]])/y21

        # TODO: add svg function
    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        return [], ((0,0), (0,0)), (0, 0)


class ABCDInductiveBranch(ABCDMatrix):
    def __init__(self, l1, l2, m12, branching_abcd):
        self.l1 = l1
        self.l2 = l2
        self.m12 = m12
        self.branching_abcd = branching_abcd

    def abcd(self, w):
        i = self.i(w)
        z_branch = self.branching_abcd.input_impedance(w, 0)
        return np.array([[i, 1j*w*self.l1 - (1j*w*self.m12)**2/(z_branch + 1j*w*self.l2)], [0*i, i]])

    def _svg_group(self, dwg, fmt, plot_labels, labels_precision):
        return self.value._svg_group(dwg, fmt, plot_labels, labels_precision)
