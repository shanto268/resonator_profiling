"""
Transmission Line Models for gdsfactory components using scikit-rf (skrf)
=========================================================================

This module shows how to convert gdsfactory transmission line geometries 
(microstrip, CPW, etc.) into skrf Network objects for RF analysis.
"""

import numpy as np
import skrf as rf
from skrf.media import MLine, CPW, DefinedGammaZ0
import gdsfactory as gf
from typing import Optional, Dict, Any


class GDSFactoryToSKRF:
    """
    Convert gdsfactory transmission line components to skrf Network objects.
    
    This class takes physical parameters from gdsfactory cells and creates
    the appropriate skrf media objects for RF simulation.
    """
    
    def __init__(self, frequency: rf.Frequency):
        """
        Initialize the converter.
        
        Parameters
        ----------
        frequency : rf.Frequency
            Frequency object defining the simulation frequency range
            Example: rf.Frequency(1, 10, 101, 'GHz')
        """
        self.freq = frequency
        
    def microstrip_from_geometry(
        self,
        width: float,
        length: float,
        substrate_height: float,
        substrate_eps_r: float = 11.9,  # Silicon
        metal_thickness: float = 0.5e-6,
        conductivity: float = 5.8e7,  # Copper
        loss_tangent: float = 0.001,
        **kwargs
    ) -> rf.Network:
        """
        Create a microstrip transmission line Network from geometry.
        
        Parameters
        ----------
        width : float
            Width of the microstrip trace (meters)
        length : float
            Length of the transmission line (meters)
        substrate_height : float
            Height of substrate between trace and ground (meters)
        substrate_eps_r : float
            Relative permittivity of substrate (11.9 for Si)
        metal_thickness : float
            Thickness of the metal trace (meters)
        conductivity : float
            Conductivity of the metal (S/m)
        loss_tangent : float
            Loss tangent of the dielectric
        
        Returns
        -------
        rf.Network
            2-port network representing the transmission line
        """
        # Create microstrip media object
        mline = MLine(
            frequency=self.freq,
            w=width,
            h=substrate_height,
            t=metal_thickness,
            ep_r=substrate_eps_r,
            mu_r=1.0,
            rho=1/conductivity,  # resistivity
            tand=loss_tangent,
            **kwargs
        )
        
        # Create transmission line network
        return mline.line(length, unit='m', name=f'microstrip_L{length*1e3:.1f}mm')
    
    def cpw_from_geometry(
        self,
        center_width: float,
        gap_width: float,
        length: float,
        substrate_height: float,
        substrate_eps_r: float = 11.9,  # Silicon
        metal_thickness: float = 0.5e-6,
        conductivity: float = 5.8e7,
        loss_tangent: float = 0.001,
        has_ground_plane: bool = False,
        **kwargs
    ) -> rf.Network:
        """
        Create a CPW transmission line Network from geometry.
        
        Parameters
        ----------
        center_width : float
            Width of the center conductor (meters)
        gap_width : float
            Gap between center conductor and ground (meters)
        length : float
            Length of the transmission line (meters)
        substrate_height : float
            Height of the substrate (meters)
        substrate_eps_r : float
            Relative permittivity of substrate
        metal_thickness : float
            Thickness of the metal (meters)
        conductivity : float
            Conductivity of the metal (S/m)
        loss_tangent : float
            Loss tangent of the dielectric
        has_ground_plane : bool
            True for conductor-backed CPW (CBCPW/CPWG)
        
        Returns
        -------
        rf.Network
            2-port network representing the CPW line
        """
        # Create CPW media object
        cpw = CPW(
            frequency=self.freq,
            w=center_width,
            s=gap_width,
            h=substrate_height,
            t=metal_thickness,
            ep_r=substrate_eps_r,
            rho=1/conductivity,
            tand=loss_tangent,
            has_metal_backside=has_ground_plane,
            **kwargs
        )
        
        # Create transmission line network
        return cpw.line(length, unit='m', name=f'cpw_L{length*1e3:.1f}mm')
    
    def from_gdsfactory_straight(
        self,
        component: gf.Component,
        line_type: str = 'microstrip',
        material_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> rf.Network:
        """
        Convert a gdsfactory straight component to skrf Network.
        
        Parameters
        ----------
        component : gf.Component
            gdsfactory component (should be a straight transmission line)
        line_type : str
            Type of transmission line ('microstrip', 'cpw', 'stripline')
        material_params : dict
            Material and substrate parameters
        
        Returns
        -------
        rf.Network
            Network representation of the transmission line
        """
        # Extract geometry from gdsfactory component
        # This is simplified - actual implementation would depend on 
        # how you store metadata in your gdsfactory components
        
        # Get bounding box to estimate length
        bbox = component.bbox()
        length = (bbox[1][0] - bbox[0][0]) * 1e-6  # Convert um to m
        
        # Default material parameters for silicon substrate
        if material_params is None:
            material_params = {
                'substrate_eps_r': 11.9,  # Silicon
                'substrate_height': 100e-6,  # 100 um
                'metal_thickness': 0.5e-6,  # 500 nm
                'conductivity': 5.8e7,  # Copper
                'loss_tangent': 0.001
            }
        
        # Get width from component settings or estimate from geometry
        # This assumes the component has settings stored
        if hasattr(component, 'settings'):
            width = component.settings.get('width', 10e-6)
        else:
            # Estimate from bounding box
            width = (bbox[1][1] - bbox[0][1]) * 1e-6
        
        if line_type == 'microstrip':
            return self.microstrip_from_geometry(
                width=width,
                length=length,
                **material_params,
                **kwargs
            )
        elif line_type == 'cpw':
            # For CPW, need gap width too
            gap_width = material_params.get('gap_width', 5e-6)
            return self.cpw_from_geometry(
                center_width=width,
                gap_width=gap_width,
                length=length,
                **material_params,
                **kwargs
            )
        else:
            raise ValueError(f"Line type {line_type} not implemented")
    
    def custom_from_propagation_constant(
        self,
        length: float,
        z0: complex,
        gamma: complex,
        name: str = 'custom_tline'
    ) -> rf.Network:
        """
        Create a transmission line from known propagation constant and Z0.
        
        This is useful when you have electromagnetic simulation results
        or measured parameters.
        
        Parameters
        ----------
        length : float
            Physical length of the line (meters)
        z0 : complex or array-like
            Characteristic impedance (can be frequency-dependent)
        gamma : complex or array-like
            Propagation constant (can be frequency-dependent)
        name : str
            Name for the network
        
        Returns
        -------
        rf.Network
            2-port network
        """
        # Create media with defined gamma and Z0
        media = DefinedGammaZ0(
            frequency=self.freq,
            z0=z0,
            gamma=gamma
        )
        
        return media.line(length, unit='m', name=name)


# Example usage functions
def example_microstrip():
    """Example: Create a microstrip line model."""
    # Define frequency range
    freq = rf.Frequency(1, 10, 101, 'GHz')
    
    # Create converter
    converter = GDSFactoryToSKRF(freq)
    
    # Create a 5mm long, 50um wide microstrip on 500um Si substrate
    network = converter.microstrip_from_geometry(
        width=50e-6,  # 50 um
        length=5e-3,  # 5 mm
        substrate_height=500e-6,  # 500 um
        substrate_eps_r=11.9,  # Silicon
        metal_thickness=1e-6,  # 1 um
    )
    
    print(f"Microstrip Z0: {network.z0[0,0]:.2f} Ohm")
    print(f"Propagation constant at 5GHz: {network.gamma[50]:.4f}")
    
    return network


def example_cpw():
    """Example: Create a CPW line model."""
    freq = rf.Frequency(1, 40, 201, 'GHz')
    converter = GDSFactoryToSKRF(freq)
    
    # Create a CPW with 30um center conductor, 20um gaps
    network = converter.cpw_from_geometry(
        center_width=30e-6,  # 30 um
        gap_width=20e-6,  # 20 um
        length=3e-3,  # 3 mm
        substrate_height=500e-6,  # 500 um Si substrate
        substrate_eps_r=11.9,
        has_ground_plane=True  # Conductor-backed CPW
    )
    
    print(f"CPW Z0: {network.z0[0,0]:.2f} Ohm")
    
    return network


def example_gdsfactory_integration():
    """Example: Convert gdsfactory component to RF network."""
    # Create a gdsfactory straight component
    # (This is a simplified example - actual integration would be more sophisticated)
    
    # Create a straight waveguide in gdsfactory
    straight = gf.components.straight(length=1000, width=10)  # 1mm long, 10um wide
    
    # Define frequency
    freq = rf.Frequency(1, 20, 101, 'GHz')
    
    # Convert to RF network
    converter = GDSFactoryToSKRF(freq)
    
    # Convert with custom material stack
    material_stack = {
        'substrate_eps_r': 11.9,  # Silicon
        'substrate_height': 200e-6,  # 200 um SOI
        'metal_thickness': 0.5e-6,  # 500 nm metal
        'conductivity': 3.7e7,  # Aluminum
        'loss_tangent': 0.001
    }
    
    network = converter.from_gdsfactory_straight(
        straight,
        line_type='microstrip',
        material_params=material_stack
    )
    
    return network


def cascade_example():
    """Example: Cascade multiple transmission line sections."""
    freq = rf.Frequency(1, 10, 51, 'GHz')
    converter = GDSFactoryToSKRF(freq)
    
    # Create different sections
    section1 = converter.microstrip_from_geometry(
        width=100e-6, length=2e-3, substrate_height=500e-6
    )
    
    section2 = converter.microstrip_from_geometry(
        width=50e-6, length=1e-3, substrate_height=500e-6
    )
    
    section3 = converter.microstrip_from_geometry(
        width=100e-6, length=2e-3, substrate_height=500e-6
    )
    
    # Cascade them (this creates a simple impedance transformer)
    cascaded = section1 ** section2 ** section3
    cascaded.name = 'impedance_transformer'
    
    return cascaded


if __name__ == "__main__":
    # Run examples
    print("=" * 60)
    print("Microstrip Example:")
    ms_network = example_microstrip()
    
    print("\n" + "=" * 60)
    print("CPW Example:")
    cpw_network = example_cpw()
    
    print("\n" + "=" * 60)
    print("Cascaded Sections Example:")
    cascaded_network = cascade_example()
    print(f"Total S21 at 5GHz: {cascaded_network.s[25,1,0]:.4f}")
