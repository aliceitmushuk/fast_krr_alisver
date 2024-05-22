def sort_data(data, sort_keys):
    # Define the custom order for the "opt" key
    opt_order = {"askotch": 0, "skotch": 1, "sketchysaga": 2, "sketchykatyusha": 3, "pcg": 4}

    # Create the sorting key function based on the specified sort_keys
    def sorting_key(d):
        d_config = d.config
        key = []
        if 'opt' in sort_keys:
            key.append(opt_order.get(d_config.get("opt"), float('inf')))
        if 'b' in sort_keys:
            key.append(d_config.get("b", float('inf')))
        if 'r' in sort_keys:
            precond_params = d_config.get("precond_params", {})
            if precond_params is None:
                precond_params = {}
            key.append(precond_params.get("r", float('inf')))
        if 'preconditioner_type' in sort_keys:
            precond_params = d_config.get("precond_params", {})
            if precond_params is None:
                precond_params = {}
            key.append(precond_params.get("type", 'zzz'))  # 'zzz' to ensure it's sorted last if not present
        return tuple(key)
    
    # Sort the data using the sorting key
    data.sort(key=sorting_key)

    return data

# def sort_data(data, sort_keys):
#     # Define the custom order for the "opt" key
#     opt_order = {"askotch": 0, "skotch": 1, "sketchysaga": 2, "sketchykatyusha": 3, "pcg": 4}

#     # Create the sorting key function based on the specified sort_keys
#     def sorting_key(d):
#         d_config = d.config
#         key = []
#         if 'opt' in sort_keys:
#             key.append(opt_order[d_config["opt"]])
#         if 'b' in sort_keys:
#             key.append(d_config.get("b", float('inf')))
#         if 'r' in sort_keys:
#             key.append(d_config.get("precond_params", {}).get("r", float('inf')))
#         if 'preconditioner_type' in sort_keys:
#             key.append(d_config.get("precond_params", {}).get("type", 'zzz'))  # 'zzz' to ensure it's sorted last if not present
#         return tuple(key)
    
#     # Sort the data using the sorting key
#     data.sort(key=sorting_key)

#     return data