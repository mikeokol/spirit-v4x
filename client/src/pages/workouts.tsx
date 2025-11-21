import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { insertWorkoutSchema, type InsertWorkout, type Workout } from "@shared/schema";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Dumbbell, Loader2, Copy, RotateCcw, History, Download, FileText } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { exportWorkoutAsPDF, exportWorkoutAsTXT } from "@/lib/export";

export default function WorkoutsPage() {
  const [selectedWorkout, setSelectedWorkout] = useState<Workout | null>(null);
  const { toast } = useToast();

  const { data: workouts = [], isLoading, isError, error } = useQuery<Workout[]>({
    queryKey: ["/api/workouts"],
  });

  const form = useForm<InsertWorkout>({
    resolver: zodResolver(insertWorkoutSchema),
    defaultValues: {
      goal: "",
      duration: "30",
      equipment: "",
      fitnessLevel: "intermediate",
    },
  });

  const generateWorkoutMutation = useMutation({
    mutationFn: async (data: InsertWorkout) => {
      const result = await apiRequest("POST", "/api/workouts/generate", data);
      return result as Workout;
    },
    onSuccess: (data) => {
      setSelectedWorkout(data);
      queryClient.invalidateQueries({ queryKey: ["/api/workouts"] });
      toast({
        title: "Workout generated!",
        description: "Your personalized workout is ready.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to generate workout. Please try again.",
        variant: "destructive",
      });
    },
  });

  const onSubmit = (data: InsertWorkout) => {
    generateWorkoutMutation.mutate(data);
  };

  const handleCopy = (workout: Workout) => {
    const text = `${workout.title}\n\n${workout.exercises
      .map(
        (ex) =>
          `${ex.name}\n${ex.sets ? `Sets: ${ex.sets}` : ""}${ex.reps ? ` | Reps: ${ex.reps}` : ""}${ex.duration ? ` | Duration: ${ex.duration}` : ""}\n${ex.notes || ""}`
      )
      .join("\n\n")}`;
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied!",
      description: "Workout copied to clipboard.",
    });
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
        <div>
          <h1 className="text-4xl font-semibold mb-2">Workout Generator</h1>
          <p className="text-muted-foreground">
            Create personalized workout plans tailored to your goals and fitness level.
          </p>
        </div>

        <Tabs defaultValue="generate" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="generate" data-testid="tab-generate">
              <Dumbbell className="h-4 w-4 mr-2" />
              Generate
            </TabsTrigger>
            <TabsTrigger value="history" data-testid="tab-history">
              <History className="h-4 w-4 mr-2" />
              History ({workouts.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="generate" className="space-y-8 mt-8">
            <Card>
              <CardHeader>
                <CardTitle>Generate Your Workout</CardTitle>
              </CardHeader>
              <CardContent>
                <Form {...form}>
                  <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={form.control}
                        name="goal"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Fitness Goal</FormLabel>
                            <FormControl>
                              <Input
                                placeholder="e.g., Build strength, lose weight, improve cardio"
                                data-testid="input-workout-goal"
                                {...field}
                              />
                            </FormControl>
                            <FormDescription>What do you want to achieve?</FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="duration"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Duration (minutes)</FormLabel>
                            <FormControl>
                              <Input
                                type="number"
                                placeholder="30"
                                data-testid="input-workout-duration"
                                {...field}
                              />
                            </FormControl>
                            <FormDescription>How long should the workout be?</FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="fitnessLevel"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Fitness Level</FormLabel>
                            <Select onValueChange={field.onChange} defaultValue={field.value}>
                              <FormControl>
                                <SelectTrigger data-testid="select-fitness-level">
                                  <SelectValue placeholder="Select your level" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="beginner">Beginner</SelectItem>
                                <SelectItem value="intermediate">Intermediate</SelectItem>
                                <SelectItem value="advanced">Advanced</SelectItem>
                              </SelectContent>
                            </Select>
                            <FormDescription>Your current fitness level</FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="equipment"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Equipment Available (Optional)</FormLabel>
                            <FormControl>
                              <Input
                                placeholder="e.g., dumbbells, resistance bands"
                                data-testid="input-workout-equipment"
                                {...field}
                              />
                            </FormControl>
                            <FormDescription>What equipment do you have?</FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </div>

                    <Button
                      type="submit"
                      className="w-full"
                      disabled={generateWorkoutMutation.isPending}
                      data-testid="button-generate-workout"
                    >
                      {generateWorkoutMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Dumbbell className="mr-2 h-5 w-5" />
                          Generate Workout
                        </>
                      )}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>

            {selectedWorkout && (
              <Card data-testid="card-generated-workout">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>{selectedWorkout.title}</CardTitle>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleCopy(selectedWorkout)}
                        data-testid="button-copy-workout"
                      >
                        <Copy className="h-4 w-4 mr-2" />
                        Copy
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => form.handleSubmit(onSubmit)()}
                        disabled={generateWorkoutMutation.isPending}
                        data-testid="button-regenerate-workout"
                      >
                        <RotateCcw className="h-4 w-4 mr-2" />
                        Regenerate
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {selectedWorkout.exercises?.map((exercise, index) => (
                      <div
                        key={index}
                        className="p-4 rounded-lg border border-card-border bg-card/50"
                        data-testid={`exercise-${index}`}
                      >
                        <h3 className="font-medium mb-2">{exercise.name}</h3>
                        <div className="grid grid-cols-3 gap-2 text-sm text-muted-foreground mb-2">
                          {exercise.sets && (
                            <div>
                              <span className="font-medium">Sets:</span> {exercise.sets}
                            </div>
                          )}
                          {exercise.reps && (
                            <div>
                              <span className="font-medium">Reps:</span> {exercise.reps}
                            </div>
                          )}
                          {exercise.duration && (
                            <div>
                              <span className="font-medium">Duration:</span> {exercise.duration}
                            </div>
                          )}
                        </div>
                        {exercise.notes && (
                          <p className="text-sm text-muted-foreground">{exercise.notes}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {!selectedWorkout && !generateWorkoutMutation.isPending && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="rounded-full bg-primary/10 p-4 mb-4">
                  <Dumbbell className="h-8 w-8 text-primary" />
                </div>
                <h2 className="text-2xl font-semibold mb-2">Ready to get started?</h2>
                <p className="text-muted-foreground max-w-md">
                  Fill in your details above and generate a personalized workout plan.
                </p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="history" className="space-y-4 mt-8">
            {isError ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="rounded-full bg-destructive/10 p-4 mb-4">
                  <History className="h-8 w-8 text-destructive" />
                </div>
                <h2 className="text-2xl font-semibold mb-2">Failed to load workouts</h2>
                <p className="text-muted-foreground max-w-md mb-4">
                  {error instanceof Error ? error.message : "An error occurred while loading your workout history."}
                </p>
                <Button
                  variant="outline"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["/api/workouts"] })}
                  data-testid="button-retry-workouts"
                >
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Try Again
                </Button>
              </div>
            ) : isLoading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <Card key={i}>
                    <CardHeader>
                      <Skeleton className="h-6 w-3/4" />
                    </CardHeader>
                    <CardContent>
                      <Skeleton className="h-20 w-full" />
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : workouts.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="rounded-full bg-primary/10 p-4 mb-4">
                  <History className="h-8 w-8 text-primary" />
                </div>
                <h2 className="text-2xl font-semibold mb-2">No workouts yet</h2>
                <p className="text-muted-foreground max-w-md">
                  Generate your first workout to see it here.
                </p>
              </div>
            ) : (
              workouts.map((workout) => (
                <Card key={workout.id} data-testid={`workout-${workout.id}`}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>{workout.title}</CardTitle>
                        <p className="text-xs text-muted-foreground mt-1">
                          {new Date(workout.timestamp).toLocaleDateString()} at{" "}
                          {new Date(workout.timestamp).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleCopy(workout)}
                          data-testid={`button-copy-${workout.id}`}
                        >
                          <Copy className="h-4 w-4 mr-2" />
                          Copy
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => exportWorkoutAsPDF(workout)}
                          data-testid={`button-export-pdf-${workout.id}`}
                        >
                          <Download className="h-4 w-4 mr-2" />
                          PDF
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => exportWorkoutAsTXT(workout)}
                          data-testid={`button-export-txt-${workout.id}`}
                        >
                          <FileText className="h-4 w-4 mr-2" />
                          TXT
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {workout.exercises?.map((exercise, index) => (
                        <div
                          key={index}
                          className="p-4 rounded-lg border border-card-border bg-card/50"
                        >
                          <h3 className="font-medium mb-2">{exercise.name}</h3>
                          <div className="grid grid-cols-3 gap-2 text-sm text-muted-foreground mb-2">
                            {exercise.sets && (
                              <div>
                                <span className="font-medium">Sets:</span> {exercise.sets}
                              </div>
                            )}
                            {exercise.reps && (
                              <div>
                                <span className="font-medium">Reps:</span> {exercise.reps}
                              </div>
                            )}
                            {exercise.duration && (
                              <div>
                                <span className="font-medium">Duration:</span> {exercise.duration}
                              </div>
                            )}
                          </div>
                          {exercise.notes && (
                            <p className="text-sm text-muted-foreground">{exercise.notes}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
