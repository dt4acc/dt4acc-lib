import at

from dt4acc_lib.pyat_simulator.multipole_proxy import MultipoleProxy


def create_at_proxy_factory():
    return {
        at.Quadrupole.__name__ :
        MultipoleProxy()
        at.Multipole.__name__,
    }
