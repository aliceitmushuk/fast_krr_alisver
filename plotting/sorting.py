from get_opt import _get_opt


def sort_data(data, sort_keys):
    # Define sorting orders
    opt_order = {
        "askotchv2": 0,
        "skotchv2": 1,
        "mimosa": 2,
        "sap": 3,
        "nsap": 4,
        "eigenpro2": 5,
        "eigenpro3": 6,
        "pcg": 7,
    }
    accelerated_order = {
        True: 0,
        False: 1,
    }
    sampling_order = {
        "uniform": 0,
        "rls": 1,
    }
    precond_order = {
        "nystrom": 0,
        "partial_cholesky": 1,
        "falkon": 3,
    }
    rho_order = {"damped": 0, "regularization": 1}
    mode_order = {"rpc": 0, "greedy": 1}

    # Create the sorting key function
    def sorting_key(d):
        d_config = d.config
        key = []

        # Sort by "opt" key
        if "opt" in sort_keys:
            key.append(opt_order.get(_get_opt(d), float("inf")))

        # Sort by "accelerated" key
        if "accelerated" in sort_keys:
            key.append(
                accelerated_order.get(d_config.get("accelerated", False), float("inf"))
            )

        # Sort by "sampling" key
        if "sampling" in sort_keys:
            key.append(
                sampling_order.get(
                    d_config.get("sampling_method", "uniform"), float("inf")
                )
            )

        # Sort by "precond_type" key
        if "precond_type" in sort_keys:
            precond_params = d_config.get("precond_params", {})
            if precond_params is None:
                precond_params = {}
            precond_type = precond_params.get("type", "zzz")

            # Handle "nystrom" specifically
            if precond_type == "nystrom":
                key.append(precond_order[precond_type])  # 0 for "nystrom"
                rho = precond_params.get("rho", float("inf"))
                if isinstance(rho, str):
                    key.append(rho_order.get(rho, float("inf")))
                elif isinstance(rho, (int, float)):
                    key.append(2 + rho)  # Offset numerical rho values
                else:
                    key.append(float("inf"))

            # Handle "partial_cholesky"
            elif precond_type == "partial_cholesky":
                key.append(precond_order[precond_type])
                mode = precond_params.get("mode", float("inf"))
                key.append(mode_order.get(mode, float("inf")))

            # Handle "falkon" or other preconditioners
            else:
                key.append(precond_order.get(precond_type, float("inf")))

        # Sort by "r" key
        if "r" in sort_keys:
            precond_params = d_config.get("precond_params", {})
            if precond_params is None:
                precond_params = {}
            key.append(precond_params.get("r", float("inf")))

        if "b" in sort_keys:
            key.append(d_config.get("block_sz", float("inf")))

        # Sort by "m" key
        if "m" in sort_keys:
            key.append(d_config.get("m", float("inf")))

        return tuple(key)

    # Sort the data
    data.sort(key=sorting_key)
    return data
