def get_provider_options(options):
    if options is None:
        return None

    provider_options = {}
    for opt in options:
        assert (
            "=" in opt
        ), "option({}) should contain a =" "to distinguish between key and value".format(opt)
        pair = opt.split("=")
        assert len(pair) == 2, "there can be only a = in the option"
        key, value = pair
        provider_options[key] = value

    return provider_options
