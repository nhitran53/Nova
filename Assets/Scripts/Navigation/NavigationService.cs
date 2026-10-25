using System;
using System.Collections.Generic;
using UnityEngine;

// Field names match navigation JSON.
[Serializable]
public class NavigationRequest
{
    public string destination_id;
    [Tooltip("Optional. Empty means any precomputed route to that destination.")]
    public string preference;
    [Tooltip("For start_navigation. Empty means the route found last.")]
    public string route_id;
}

[Serializable]
public class NavigationResponse
{
    public bool success;
    public string route_id;
    public string destination_id;
    public string preference;
    public float distance_meters;
    public float estimated_time_seconds;
    public string route_type;
    public string status;
}

public class NavigationService
{
    readonly IReadOnlyList<NavigationRoute> routes;
    readonly Func<float> walkingSpeed;

    public NavigationService(IReadOnlyList<NavigationRoute> routes, Func<float> walkingSpeed)
    {
        this.routes = routes;
        this.walkingSpeed = walkingSpeed;
    }

    public IReadOnlyList<NavigationRoute> Routes => routes;

    public NavigationResponse FindNavigationRoute(NavigationRequest request, out NavigationRoute match)
    {
        match = null;
        var response = new NavigationResponse { route_type = "precomputed", destination_id = request?.destination_id };
        if (request == null || string.IsNullOrWhiteSpace(request.destination_id))
            return Fail(response, "error: destination_id is required");

        var known = new List<string>();
        NavigationRoute fallback = null;
        foreach (var route in routes)
        {
            if (route == null) continue;
            known.Add(route.destinationId);
            if (!string.Equals(route.destinationId, request.destination_id, StringComparison.OrdinalIgnoreCase)) continue;
            if (fallback == null) fallback = route;
            if (string.IsNullOrWhiteSpace(request.preference) ||
                string.Equals(route.preference, request.preference, StringComparison.OrdinalIgnoreCase))
            {
                match = route;
                break;
            }
        }

        // Report when falling back to another route preference.
        string note = null;
        if (match == null && fallback != null)
        {
            match = fallback;
            note = $"no separate '{request.preference}' route; returning the precomputed {fallback.preference} route";
        }
        if (match == null)
            return Fail(response, $"error: no precomputed route to {request.destination_id}. Known destinations: {(known.Count > 0 ? string.Join(", ", known) : "none")}");
        if (!Fill(response, match))
        {
            match = null;
            return response;
        }
        response.status = note == null ? "found" : "found: " + note;
        return response;
    }

    public NavigationResponse GetRoute(string routeId, out NavigationRoute match)
    {
        match = null;
        var response = new NavigationResponse { route_type = "precomputed", route_id = routeId };
        foreach (var route in routes)
            if (route != null && string.Equals(route.routeId, routeId, StringComparison.OrdinalIgnoreCase)) match = route;
        if (match == null) return Fail(response, $"error: no precomputed route with id '{routeId}'");
        if (!Fill(response, match)) match = null;
        return response;
    }

    public NavigationResponse Describe(NavigationRoute route, string status)
    {
        var response = new NavigationResponse { route_type = "precomputed" };
        if (route != null) Fill(response, route);
        response.success = route != null;
        response.status = status;
        return response;
    }

    bool Fill(NavigationResponse response, NavigationRoute route)
    {
        response.route_id = route.routeId;
        response.destination_id = route.destinationId;
        response.preference = route.preference;
        if (!route.IsBuilt)
        {
            Fail(response, $"error: route {route.routeId} has no points yet. Run Tools > Nova > Navigation > Rebuild Shuttle Route");
            return false;
        }
        response.success = true;
        response.distance_meters = Mathf.Round(route.lengthMeters * 10) / 10;
        response.estimated_time_seconds = Mathf.Round(route.lengthMeters / Mathf.Max(0.1f, walkingSpeed()));
        return true;
    }

    static NavigationResponse Fail(NavigationResponse response, string status)
    {
        response.success = false;
        response.status = status;
        return response;
    }
}
