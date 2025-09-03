"""Convert gdsfactory components to skrf Networks for RF analysis."""

import skrf as rf
from skrf.media import MLine, CPW, DefinedGammaZ0
import gdsfactory as gf
from typing import Union
from async_lru import alru_cache
import asyncio


@alru_cache
async def to_skrf(
    component: gf.Component,
    cross_section: gf.CrossSection,
    freq: rf.Frequency,
    **kwargs
) -> rf.Network:
    """
    Convert gdsfactory component to skrf Network.
    
    Parameters
    ----------
    component : gf.Component
        gdsfactory component (typically a straight)
    cross_section : gf.CrossSection
        Cross-section containing layer stack info
    freq : rf.Frequency
        Frequency range for analysis
    **kwargs : dict
        Additional parameters (eps_r, height, conductivity, etc.)
    
    Returns
    -------
    rf.Network
        2-port transmission line network
    """
    # Extract length from component
    bbox = component.bbox()
    length = (bbox[1][0] - bbox[0][0]) * 1e-6  # um to m
    
    # Get width from cross-section
    width = cross_section.width * 1e-6 if cross_section.width else 10e-6
    
    # Determine line type from cross-section
    if hasattr(cross_section, 'gap') and cross_section.gap:
        network = await _make_cpw(width, cross_section.gap * 1e-6, length, freq, **kwargs)
    else:
        network = await _make_microstrip(width, length, freq, **kwargs)
    
    return network


@alru_cache
async def _make_microstrip(
    width: float,
    length: float, 
    freq: rf.Frequency,
    eps_r: float = 11.9,  # Si
    height: float = 500e-6,
    thickness: float = 1e-6,
    rho: float = 1.68e-8,  # Cu resistivity
    tand: float = 0.001,
    **kwargs
) -> rf.Network:
    """Create microstrip Network."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: MLine(
            frequency=freq,
            w=width,
            h=height,
            t=thickness,
            ep_r=eps_r,
            rho=rho,
            tand=tand,
            **kwargs
        ).line(length, unit='m')
    )


@alru_cache
async def _make_cpw(
    width: float,
    gap: float,
    length: float,
    freq: rf.Frequency,
    eps_r: float = 11.9,
    height: float = 500e-6,
    thickness: float = 1e-6,
    rho: float = 1.68e-8,
    tand: float = 0.001,
    grounded: bool = False,
    **kwargs
) -> rf.Network:
    """Create CPW Network."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: CPW(
            frequency=freq,
            w=width,
            s=gap,
            h=height,
            t=thickness,
            ep_r=eps_r,
            rho=rho,
            tand=tand,
            has_metal_backside=grounded,
            **kwargs
        ).line(length, unit='m')
    )


@alru_cache
async def from_em_sim(
    component: gf.Component,
    z0: Union[float, complex],
    gamma: Union[float, complex],
    freq: rf.Frequency,
) -> rf.Network:
    """
    Create Network from EM simulation results.
    
    Parameters
    ----------
    component : gf.Component
        Component for extracting length
    z0 : float or complex
        Characteristic impedance
    gamma : float or complex  
        Propagation constant
    freq : rf.Frequency
        Frequency range
    
    Returns
    -------
    rf.Network
        Transmission line network
    """
    bbox = component.bbox()
    length = (bbox[1][0] - bbox[0][0]) * 1e-6
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: DefinedGammaZ0(frequency=freq, z0=z0, gamma=gamma).line(length, unit='m')
    )


# Example usage
async def main():
    """Example async usage."""
    freq = rf.Frequency(1, 40, 201, 'GHz')
    c = gf.components.straight(length=1000)
    xs = gf.cross_section.strip(width=10)
    
    network = await to_skrf(c, xs, freq, eps_r=11.9, height=200e-6)
    
    print(f"Z0: {network.z0[0,0]:.1f} Ω")
    print(f"S21 @ 20GHz: {abs(network.s[100,1,0]):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
