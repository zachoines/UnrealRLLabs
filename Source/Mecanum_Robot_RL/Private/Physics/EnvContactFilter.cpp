// Implementation of per-world Chaos contact filter that disables contacts
// between bodies belonging to different environment instances.

#include "Physics/EnvContactFilter.h"
// Use LogTemp to avoid depending on module log header

#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "Components/PrimitiveComponent.h"

#if WITH_CHAOS && UNREALRLLABS_ENABLE_CHAOS_CONTACT_FILTER
#include "Physics/Experimental/PhysScene_Chaos.h"
// The following Chaos includes may vary across engine minor versions.
// If these paths do not resolve for your UE version, adjust accordingly.
// Contact modification interface
#include "Chaos/ContactModification.h"
// User data helpers to retrieve owning components from particles
#include "Chaos/ChaosUserData.h"
#endif

namespace
{
    // We encode the environment id as a component tag in form "EnvId=<n>".
    // This helper parses such tags from a component or its owner actor.
    static int32 GetEnvIdFromComponentTags(const UPrimitiveComponent* Comp)
    {
        if (!Comp) return -1;
        auto ParseTags = [](const TArray<FName>& Tags) -> int32
        {
            for (const FName& Tag : Tags)
            {
                const FString S = Tag.ToString();
                if (S.StartsWith(TEXT("EnvId=")))
                {
                    int32 V = -1;
                    const FString Tail = S.RightChop(6);
                    if (Tail.Len() > 0)
                    {
                        LexFromString(V, *Tail);
                        return V;
                    }
                }
            }
            return -1;
        };

        int32 Id = ParseTags(Comp->ComponentTags);
        if (Id >= 0) return Id;
        if (const AActor* Owner = Comp->GetOwner())
        {
            Id = ParseTags(Owner->Tags);
        }
        return Id;
    }

#if WITH_CHAOS && UNREALRLLABS_ENABLE_CHAOS_CONTACT_FILTER
    // Wrapper that pulls a UPrimitiveComponent* from a Chaos particle's user data.
    static const UPrimitiveComponent* GetComponentFromParticle(const Chaos::FGeometryParticle* Particle)
    {
        if (!Particle) return nullptr;
        const void* UD = Particle->UserData();
        if (!UD) return nullptr;
        // FChaosUserData lets us retrieve the owning component set by the engine.
        if (const UPrimitiveComponent* Comp = Chaos::FChaosUserData::Get<UPrimitiveComponent>(UD))
        {
            return Comp;
        }
        // Some engine versions store FBodyInstance instead; try to recover component from it.
        if (const FBodyInstance* BI = Chaos::FChaosUserData::Get<FBodyInstance>(UD))
        {
            return BI->GetPrimitiveComponent();
        }
        return nullptr;
    }

    // Contact modifier that disables contacts across different EnvIds
    class FEnvContactModifier final : public Chaos::IContactModifier
    {
    public:
        virtual void Modify(Chaos::FCollisionContactModifier& Modifier) override
        {
            // Iterate contact pairs and veto cross-env contacts
            for (Chaos::FContactPairModifier& Pair : Modifier)
            {
                const Chaos::FGeometryParticle* P0 = Pair.GetParticle(0);
                const Chaos::FGeometryParticle* P1 = Pair.GetParticle(1);

                const UPrimitiveComponent* C0 = GetComponentFromParticle(P0);
                const UPrimitiveComponent* C1 = GetComponentFromParticle(P1);

                const int32 E0 = GetEnvIdFromComponentTags(C0);
                const int32 E1 = GetEnvIdFromComponentTags(C1);

                if (E0 >= 0 && E1 >= 0 && E0 != E1)
                {
                    // API name varies by engine version; prefer SetDisabled(true) if available.
                    // If your version uses SetEnabled(false), adjust here accordingly.
                    Pair.SetDisabled(true);
                }
            }
        }
    };
#endif // WITH_CHAOS && UNREALRLLABS_ENABLE_CHAOS_CONTACT_FILTER

    struct FEnvWorldState
    {
#if WITH_CHAOS && UNREALRLLABS_ENABLE_CHAOS_CONTACT_FILTER
        TUniquePtr<Chaos::IContactModifier> Modifier;
        Chaos::FPhysicsSolver* Solver = nullptr;
#endif
    };

    static TMap<TWeakObjectPtr<UWorld>, FEnvWorldState> GEnvFilterStates;
}

void FEnvContactFilter::Register(UWorld* World)
{
    if (!World)
    {
        return;
    }

#if WITH_CHAOS && UNREALRLLABS_ENABLE_CHAOS_CONTACT_FILTER
    FEnvWorldState& State = GEnvFilterStates.FindOrAdd(World);
    if (State.Modifier.IsValid())
    {
        return; // already registered
    }

    FPhysScene* Scene = World->GetPhysicsScene();
    if (!Scene)
    {
    UE_LOG(LogTemp, Warning, TEXT("EnvContactFilter: No physics scene for world %s"), *World->GetName());
        return;
    }

    FPhysScene_Chaos* ChaosScene = Scene->GetPhysScene_Chaos();
    if (!ChaosScene)
    {
    UE_LOG(LogTemp, Warning, TEXT("EnvContactFilter: No Chaos scene for world %s"), *World->GetName());
        return;
    }

    Chaos::FPhysicsSolver* Solver = ChaosScene->GetSolver();
    if (!Solver)
    {
    UE_LOG(LogTemp, Warning, TEXT("EnvContactFilter: No Chaos solver for world %s"), *World->GetName());
        return;
    }

    State.Modifier = MakeUnique<FEnvContactModifier>();
    State.Solver = Solver;

    // Note: The exact API to register a contact modifier can vary with UE version.
    // In recent UE versions, the solver exposes Add/RemoveContactModifier.
    // If this does not compile in your setup, replace these with the correct calls for your engine.
    Solver->AddContactModifier(State.Modifier.Get());

    UE_LOG(LogTemp, Log, TEXT("EnvContactFilter: Contact modifier installed for %s"), *World->GetName());
#else
    UE_LOG(LogTemp, Warning, TEXT("EnvContactFilter: Chaos contact modifier not enabled; define UNREALRLLABS_ENABLE_CHAOS_CONTACT_FILTER=1 to enable (and ensure Chaos headers are available)."));
#endif
}

void FEnvContactFilter::Unregister(UWorld* World)
{
    if (!World)
    {
        return;
    }

#if WITH_CHAOS && UNREALRLLABS_ENABLE_CHAOS_CONTACT_FILTER
    if (FEnvWorldState* State = GEnvFilterStates.Find(World))
    {
        if (State->Solver && State->Modifier.IsValid())
        {
            // See note in Register about API differences.
            State->Solver->RemoveContactModifier(State->Modifier.Get());
            State->Modifier.Reset();
            State->Solver = nullptr;
        }
        GEnvFilterStates.Remove(World);
    }
#endif
}
