#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Column.generated.h"

UCLASS()
class UNREALRLLABS_API AColumn : public AActor
{
    GENERATED_BODY()

public:
    // Constructor
    AColumn();

    // Initialize the column with its size, position, and maximum height
    void InitColumn(FVector Dimensions, FVector Location, float MaxHeight);

    // Set the column's height as a percentage of MaxHeight
    void SetColumnHeight(float NewHeight);

    // Get the current height as a percentage of MaxHeight
    float GetColumnHeight() const;

    // Set the color of the column
    void SetColumnColor(FLinearColor Color);

    // Set the acceleration of the column (modifies velocity)
    void SetColumnAcceleration(float Acceleration);

    // Reset the column to its initial state
    void ResetColumn();

protected:
    // Called when the game starts or when the actor is spawned
    virtual void BeginPlay() override;

public:
    // Called every frame to update the column's position based on velocity
    virtual void Tick(float DeltaTime) override;

private:
    // The static mesh component representing the column
    UPROPERTY(VisibleAnywhere)
    UStaticMeshComponent* ColumnMesh;

    // The maximum height the column can reach
    float MaximumHeight;

    // The current velocity of the column
    float Velocity;
};
