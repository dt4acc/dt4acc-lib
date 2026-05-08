Interacting with the accelerator toolbox
========================================

Proxying requests to the element
--------------------------------

Providing the backend
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: dt4acc_lib.pyat_simulator.simulator_backend
   :members:
   :undoc-members:
   :show-inheritance:


Towards an element access interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: dt4acc_lib.pyat_simulator.accelerator_simulator
   :members:
   :undoc-members:
   :show-inheritance:


The proxy itself: providing functionality in accordance to the interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dt4acc_lib.pyat_simulator.proxies.properties_proxy
   :members:
   :undoc-members:
   :show-inheritance:


Proxy factory: creating a dedicated proxy for the element in question
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is used by the accelerator simulator

.. automodule:: dt4acc_lib.pyat_simulator.proxies.proxy_factory
   :members:
   :undoc-members:
   :show-inheritance:



Changing element properties
---------------------------

Geometric properties
~~~~~~~~~~~~~~~~~~~~
.. automodule:: dt4acc_lib.pyat_simulator.element_properties.geometric_properties
   :members:
   :undoc-members:
   :show-inheritance:


Changing multipole components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: dt4acc_lib.pyat_simulator.element_properties.multipoles
   :members:
   :undoc-members:
   :show-inheritance:

Changing main strength of a magnet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: dt4acc_lib.pyat_simulator.element_properties.main_strength
   :members:
   :undoc-members:
   :show-inheritance:


The interface definition
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: dt4acc_lib.pyat_simulator.element_properties.element_property_interface
   :members:
   :undoc-members:
   :show-inheritance:


Internal: states of the calculation engine
------------------------------------------

.. automodule:: dt4acc_lib.pyat_simulator.model.calculation_states
   :members:
   :undoc-members:
   :show-inheritance:
