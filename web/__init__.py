"""Optional Flask interface; the core package never imports Flask."""


def create_app(config=None):
    from .app import create_app as factory

    return factory(config)
