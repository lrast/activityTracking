# Adds context sensitive perturbations to a model

def activate_model_tracking(model, layers_to_track, callback, track_inputs=False):
    """ Adds tracking hooks to the model """
    modules = dict(model.named_modules())

    active_hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if track_inputs:
                callback(name, output, input[0])
            else:
                callback(name, output)

        return hook

    for name in layers_to_track:

        hook = make_hook(name)
        handle = modules[name].register_forward_hook(hook)
        active_hooks.append(handle)

    return active_hooks


def clear_hooks(active_hooks):
    """ Remove all listed hooks """
    for hook in active_hooks:
        hook.remove()
