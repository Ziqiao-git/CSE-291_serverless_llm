import random
import heapq
import statistics

def simulate_queue(
    num_requests=500,
    num_models=10,
    arrival_rate=1.0,    # parameter for exponential inter-arrival distribution
    switch_time=10.0,     # x seconds to switch models
    service_time=5.0,    # y seconds to serve one request
    seed=42
):
    """
    Simulates a queueing system where only one model can be loaded at a time, switching
    models takes `switch_time`, and serving a single request takes `service_time`.
    
    Arrivals:
        - Poisson process with parameter 'arrival_rate'.
        - Each request randomly assigned to one of `num_models` with equal probability.
        
    Returns:
        average_wait (float): Average waiting time of all requests.
    """
    random.seed(seed)
    
    # 1) Generate all arrivals in advance (for simplicity, though one can also do "online" generation).
    arrival_times = []
    current_time = 0.0
    for i in range(num_requests):
        # exponential inter-arrival time
        inter_arrival = random.expovariate(arrival_rate)
        current_time += inter_arrival
        # random model for this request
        model_required = random.randint(0, num_models - 1)
        arrival_times.append( (current_time, model_required, i) )
    # Sort by arrival time (just to be safe if we had any random anomalies)
    arrival_times.sort(key=lambda x: x[0])

    # We will store events in a min-heap, where each event is:
    # (event_time, event_type, info...)
    # event_type can be:
    #   0: arrival
    #   1: service completion
    #   2: model switch completion
    
    # Preload all arrival events
    events = []
    for arr_t, mod, idx in arrival_times:
        heapq.heappush(events, (arr_t, 0, mod, idx))
    
    # We maintain a queue of waiting requests for each model
    queues = [[] for _ in range(num_models)]
    
    # We'll track the waiting times
    # waiting_time[i] = how long request i waited in queue before starting service
    waiting_time = [0.0] * num_requests
    
    # Simulation state
    current_time = 0.0
    current_model = None   # which model is currently loaded on GPU
    gpu_busy_until = 0.0   # when does the GPU become free (no more service)?
    switching_until = 0.0  # if we are switching, when will that finish?
    
    # For data structure convenience:
    #   "in_service" indicates if the GPU is actively serving requests (True/False)
    #   "in_switch" indicates if the GPU is in the middle of switching
    in_service = False
    in_switch  = False
    
    while events:
        event_time, event_type, *info = heapq.heappop(events)
        current_time = event_time
        
        if event_type == 0:
            # ARRIVAL event
            model_required, request_id = info
            # Put request in the corresponding queue
            queues[model_required].append( (request_id, current_time) )
            
            # Possibly start serving if GPU is idle and not switching
            if not in_service and not in_switch:
                # Decide what to load if current_model is None or if no requests match the current_model
                if current_model is None:
                    # pick the model with the largest queue
                    best_model = pick_best_model(queues)
                    if best_model is not None:
                        # Start switching immediately from "no model" to best_model
                        current_model = best_model
                        in_switch = True
                        switching_until = current_time + switch_time
                        # schedule switch completion event
                        heapq.heappush(events, (switching_until, 2, best_model))
                else:
                    # If we already have a model loaded, check if there's anything to serve
                    if len(queues[current_model]) > 0:
                        # serve the next request
                        serve_next_request(
                            queues, current_model, current_time, service_time,
                            waiting_time, events
                        )
                        in_service = True
                        gpu_busy_until = current_time + service_time
                    else:
                        # no requests for current_model, maybe switch to the largest queue
                        best_model = pick_best_model(queues)
                        if best_model is not None and best_model != current_model and len(queues[best_model]) > 0:
                            # switch
                            in_switch = True
                            switching_until = current_time + switch_time
                            current_model = best_model
                            heapq.heappush(events, (switching_until, 2, best_model))

        elif event_type == 1:
            # SERVICE COMPLETION event
            in_service = False
            # After finishing service, check if there is more work for the same model
            if current_model is not None and len(queues[current_model]) > 0:
                # Start next request immediately
                serve_next_request(
                    queues, current_model, current_time, service_time,
                    waiting_time, events
                )
                in_service = True
                gpu_busy_until = current_time + service_time
            else:
                # Possibly switch if there's a bigger queue for some other model
                best_model = pick_best_model(queues)
                if best_model is not None and best_model != current_model and len(queues[best_model]) > 0:
                    in_switch = True
                    switching_until = current_time + switch_time
                    current_model = best_model
                    heapq.heappush(events, (switching_until, 2, best_model))
                else:
                    # If no switching needed and no requests to serve, do nothing (idle)
                    pass

        elif event_type == 2:
            # MODEL SWITCH COMPLETION event
            in_switch = False
            loaded_model = info[0]
            # Right after switching finishes, if there are requests for the loaded model, serve them
            if len(queues[loaded_model]) > 0:
                serve_next_request(
                    queues, loaded_model, current_time, service_time,
                    waiting_time, events
                )
                in_service = True
                gpu_busy_until = current_time + service_time
            else:
                # If somehow no request is there for that model, remain idle
                pass

    # end of simulation
    average_wait = statistics.mean(waiting_time)
    return average_wait

def pick_best_model(queues):
    """
    Returns the index of the model that currently has the largest queue.
    If all queues are empty, returns None.
    """
    max_len = 0
    best_model = None
    for m, q in enumerate(queues):
        if len(q) > max_len:
            max_len = len(q)
            best_model = m
    return best_model

def serve_next_request(queues, model, current_time, service_time, waiting_time, events):
    """
    Starts service for the *oldest* waiting request in the queue of 'model'.
    Schedules a 'service completion' event.
    Updates waiting_time for that request.
    """
    # pop the earliest arrival from the queue
    request_id, arrival_t = queues[model].pop(0)
    # waiting = start_service_time - arrival_time
    waiting_time[request_id] = current_time - arrival_t
    # schedule service completion
    finish_time = current_time + service_time
    heapq.heappush(events, (finish_time, 1))

if __name__ == "__main__":
    avg_wait = simulate_queue()
    print(f"Average waiting time = {avg_wait:.3f} seconds")

