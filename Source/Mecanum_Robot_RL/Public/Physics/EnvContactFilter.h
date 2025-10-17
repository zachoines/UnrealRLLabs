// Minimal contact-filter registration facade for Chaos
// If Chaos contact modification API is available in your UE version,
// FEnvContactFilter will register a per-world modifier to veto cross-env contacts.

#pragma once

#include "CoreMinimal.h"

class UWorld;

class FEnvContactFilter
{
public:
    // Attempt to register a contact filter for the given world
    static void Register(UWorld* World);
    // Unregister previously registered filter for the given world (if any)
    static void Unregister(UWorld* World);
};

