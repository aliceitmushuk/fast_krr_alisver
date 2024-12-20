def sort_data(data, sort_keys):
    # Define the custom order for the "opt" key
    opt_order = {
        "askotchv2": 0,
        "mimosa": 1,
        "pcg": 2,
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
        "partial_cholesky": {"rpc": 1, "greedy": 2},
        "falkon": 3,
    }

    # Create the sorting key function based on the specified sort_keys
    def sorting_key(d):
        d_config = d.config
        key = []
        if "opt" in sort_keys:
            key.append(opt_order.get(d_config.get("opt"), float("inf")))
        if "accelerated" in sort_keys:
            key.append(
                accelerated_order.get(d_config.get("accelerated", False), float("inf"))
            )
        if "sampling" in sort_keys:
            key.append(
                sampling_order.get(d_config.get("sampling", "uniform"), float("inf"))
            )
        # if "b" in sort_keys:
        #     key.append(d_config.get("b", float("inf")))
        if "precond_type" in sort_keys:
            precond_params = d_config.get("precond_params", {})
            if precond_params is None:
                precond_params = {}
            precond_type = precond_params.get("type", "zzz")
            if precond_type == "partial_cholesky":
                key.append(precond_order[precond_type][precond_params.get("mode")])
            else:
                key.append(
                    precond_order.get(precond_params.get("type", "zzz"), float("inf"))
                )  # 'zzz' to ensure it's sorted last if not present
        if "r" in sort_keys:
            precond_params = d_config.get("precond_params", {})
            if precond_params is None:
                precond_params = {}
            key.append(precond_params.get("r", float("inf")))
        if "m" in sort_keys:
            key.append(d_config.get("m", float("inf")))
        return tuple(key)

    # Sort the data using the sorting key
    data.sort(key=sorting_key)

    return data
