import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { insertReflectionSchema, type InsertReflection, type Reflection } from "@shared/schema";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
import { Sparkles, Loader2, Copy, RotateCcw, History, Download, FileText } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { exportReflectionAsPDF, exportReflectionAsTXT } from "@/lib/export";

export default function ReflectionsPage() {
  const [selectedReflection, setSelectedReflection] = useState<Reflection | null>(null);
  const { toast } = useToast();

  const { data: reflections = [], isLoading, isError, error } = useQuery<Reflection[]>({
    queryKey: ["/api/reflections"],
  });

  const form = useForm<InsertReflection>({
    resolver: zodResolver(insertReflectionSchema),
    defaultValues: {
      period: "",
      focus: "",
    },
  });

  const generateReflectionMutation = useMutation({
    mutationFn: async (data: InsertReflection) => {
      const result = await apiRequest("POST", "/api/reflections/generate", data);
      return result as Reflection;
    },
    onSuccess: (data) => {
      setSelectedReflection(data);
      queryClient.invalidateQueries({ queryKey: ["/api/reflections"] });
      toast({
        title: "Reflection generated!",
        description: "Your reflection is ready for review.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to generate reflection. Please try again.",
        variant: "destructive",
      });
    },
  });

  const onSubmit = (data: InsertReflection) => {
    generateReflectionMutation.mutate(data);
  };

  const handleCopy = (reflection: Reflection) => {
    const text = `${reflection.title}\n\nWins:\n${reflection.wins.join("\n")}\n\nChallenges:\n${reflection.challenges.join("\n")}\n\nGrowth:\n${reflection.growth.join("\n")}`;
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied!",
      description: "Reflection copied to clipboard.",
    });
  };

  const renderReflectionContent = (reflection: Reflection) => (
    <div className="space-y-6">
      {reflection.tags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {reflection.tags.map((tag, index) => (
            <Badge key={index} variant="secondary" data-testid={`tag-${index}`}>
              {tag}
            </Badge>
          ))}
        </div>
      )}

      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-medium mb-3 text-green-600 dark:text-green-400">
            Wins
          </h3>
          <ul className="space-y-2" data-testid="list-wins">
            {reflection.wins.map((win, index) => (
              <li
                key={index}
                className="flex gap-2 text-sm p-3 rounded-lg bg-card/50 border border-card-border"
              >
                <span className="text-green-600 dark:text-green-400">•</span>
                <span>{win}</span>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h3 className="text-lg font-medium mb-3 text-orange-600 dark:text-orange-400">
            Challenges
          </h3>
          <ul className="space-y-2" data-testid="list-challenges">
            {reflection.challenges.map((challenge, index) => (
              <li
                key={index}
                className="flex gap-2 text-sm p-3 rounded-lg bg-card/50 border border-card-border"
              >
                <span className="text-orange-600 dark:text-orange-400">•</span>
                <span>{challenge}</span>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h3 className="text-lg font-medium mb-3 text-blue-600 dark:text-blue-400">
            Growth
          </h3>
          <ul className="space-y-2" data-testid="list-growth">
            {reflection.growth.map((item, index) => (
              <li
                key={index}
                className="flex gap-2 text-sm p-3 rounded-lg bg-card/50 border border-card-border"
              >
                <span className="text-blue-600 dark:text-blue-400">•</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
        <div>
          <h1 className="text-4xl font-semibold mb-2">Reflections</h1>
          <p className="text-muted-foreground">
            Generate guided reflections to track your progress and personal growth.
          </p>
        </div>

        <Tabs defaultValue="generate" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="generate" data-testid="tab-generate">
              <Sparkles className="h-4 w-4 mr-2" />
              Generate
            </TabsTrigger>
            <TabsTrigger value="history" data-testid="tab-history">
              <History className="h-4 w-4 mr-2" />
              History ({reflections.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="generate" className="space-y-8 mt-8">
            <Card>
              <CardHeader>
                <CardTitle>Generate Your Reflection</CardTitle>
              </CardHeader>
              <CardContent>
                <Form {...form}>
                  <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={form.control}
                        name="period"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Time Period</FormLabel>
                            <FormControl>
                              <Input
                                placeholder="e.g., This week, Last month, Q1 2025"
                                data-testid="input-reflection-period"
                                {...field}
                              />
                            </FormControl>
                            <FormDescription>What period are you reflecting on?</FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="focus"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Focus Area (Optional)</FormLabel>
                            <FormControl>
                              <Input
                                placeholder="e.g., Career, Health, Learning"
                                data-testid="input-reflection-focus"
                                {...field}
                              />
                            </FormControl>
                            <FormDescription>Specific area to reflect on</FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </div>

                    <Button
                      type="submit"
                      className="w-full"
                      disabled={generateReflectionMutation.isPending}
                      data-testid="button-generate-reflection"
                    >
                      {generateReflectionMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Sparkles className="mr-2 h-5 w-5" />
                          Generate Reflection
                        </>
                      )}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>

            {selectedReflection && (
              <Card data-testid="card-generated-reflection">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>{selectedReflection.title}</CardTitle>
                      <p className="text-xs text-muted-foreground mt-1">
                        {new Date(selectedReflection.timestamp).toLocaleDateString()}
                      </p>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleCopy(selectedReflection)}
                        data-testid="button-copy-reflection"
                      >
                        <Copy className="h-4 w-4 mr-2" />
                        Copy
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => form.handleSubmit(onSubmit)()}
                        disabled={generateReflectionMutation.isPending}
                        data-testid="button-regenerate-reflection"
                      >
                        <RotateCcw className="h-4 w-4 mr-2" />
                        Regenerate
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>{renderReflectionContent(selectedReflection)}</CardContent>
              </Card>
            )}

            {!selectedReflection && !generateReflectionMutation.isPending && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="rounded-full bg-primary/10 p-4 mb-4">
                  <Sparkles className="h-8 w-8 text-primary" />
                </div>
                <h2 className="text-2xl font-semibold mb-2">Take a moment to reflect</h2>
                <p className="text-muted-foreground max-w-md">
                  Fill in the time period above and generate a guided reflection to track your journey.
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
                <h2 className="text-2xl font-semibold mb-2">Failed to load reflections</h2>
                <p className="text-muted-foreground max-w-md mb-4">
                  {error instanceof Error ? error.message : "An error occurred while loading your reflection history."}
                </p>
                <Button
                  variant="outline"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["/api/reflections"] })}
                  data-testid="button-retry-reflections"
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
            ) : reflections.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="rounded-full bg-primary/10 p-4 mb-4">
                  <History className="h-8 w-8 text-primary" />
                </div>
                <h2 className="text-2xl font-semibold mb-2">No reflections yet</h2>
                <p className="text-muted-foreground max-w-md">
                  Generate your first reflection to see it here.
                </p>
              </div>
            ) : (
              reflections.map((reflection) => (
                <Card key={reflection.id} data-testid={`reflection-${reflection.id}`}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>{reflection.title}</CardTitle>
                        <p className="text-xs text-muted-foreground mt-1">
                          {new Date(reflection.timestamp).toLocaleDateString()} at{" "}
                          {new Date(reflection.timestamp).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleCopy(reflection)}
                          data-testid={`button-copy-${reflection.id}`}
                        >
                          <Copy className="h-4 w-4 mr-2" />
                          Copy
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => exportReflectionAsPDF(reflection)}
                          data-testid={`button-export-pdf-${reflection.id}`}
                        >
                          <Download className="h-4 w-4 mr-2" />
                          PDF
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => exportReflectionAsTXT(reflection)}
                          data-testid={`button-export-txt-${reflection.id}`}
                        >
                          <FileText className="h-4 w-4 mr-2" />
                          TXT
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>{renderReflectionContent(reflection)}</CardContent>
                </Card>
              ))
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
