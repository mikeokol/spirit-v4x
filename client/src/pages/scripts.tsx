import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { insertScriptSchema, type InsertScript, type Script } from "@shared/schema";
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
import { FileText, Loader2, Copy, RotateCcw, History, Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { exportScriptAsPDF, exportScriptAsTXT } from "@/lib/export";

export default function ScriptsPage() {
  const [selectedScript, setSelectedScript] = useState<Script | null>(null);
  const { toast } = useToast();

  const { data: scripts = [], isLoading, isError, error } = useQuery<Script[]>({
    queryKey: ["/api/scripts"],
  });

  const form = useForm<InsertScript>({
    resolver: zodResolver(insertScriptSchema),
    defaultValues: {
      topic: "",
      scriptType: "video",
      duration: "5",
      tone: "professional",
    },
  });

  const generateScriptMutation = useMutation({
    mutationFn: async (data: InsertScript) => {
      const result = await apiRequest("POST", "/api/scripts/generate", data);
      return result as Script;
    },
    onSuccess: (data) => {
      setSelectedScript(data);
      queryClient.invalidateQueries({ queryKey: ["/api/scripts"] });
      toast({
        title: "Script generated!",
        description: "Your script is ready to use.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to generate script. Please try again.",
        variant: "destructive",
      });
    },
  });

  const onSubmit = (data: InsertScript) => {
    generateScriptMutation.mutate(data);
  };

  const handleCopy = (script: Script) => {
    navigator.clipboard.writeText(script.content);
    toast({
      title: "Copied!",
      description: "Script copied to clipboard.",
    });
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-4 py-8 space-y-8">
        <div>
          <h1 className="text-4xl font-semibold mb-2">Creator Scripts</h1>
          <p className="text-muted-foreground">
            Generate professional scripts for videos, podcasts, presentations, and social media.
          </p>
        </div>

        <Tabs defaultValue="generate" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="generate" data-testid="tab-generate">
              <FileText className="h-4 w-4 mr-2" />
              Generate
            </TabsTrigger>
            <TabsTrigger value="history" data-testid="tab-history">
              <History className="h-4 w-4 mr-2" />
              History ({scripts.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="generate" className="space-y-8 mt-8">
            <Card>
              <CardHeader>
                <CardTitle>Generate Your Script</CardTitle>
              </CardHeader>
              <CardContent>
                <Form {...form}>
                  <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <FormField
                        control={form.control}
                        name="topic"
                        render={({ field }) => (
                          <FormItem className="md:col-span-2">
                            <FormLabel>Topic</FormLabel>
                            <FormControl>
                              <Input
                                placeholder="e.g., The future of AI in education"
                                data-testid="input-script-topic"
                                {...field}
                              />
                            </FormControl>
                            <FormDescription>What is your script about?</FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="scriptType"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Script Type</FormLabel>
                            <Select onValueChange={field.onChange} defaultValue={field.value}>
                              <FormControl>
                                <SelectTrigger data-testid="select-script-type">
                                  <SelectValue placeholder="Select type" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="video">Video</SelectItem>
                                <SelectItem value="podcast">Podcast</SelectItem>
                                <SelectItem value="presentation">Presentation</SelectItem>
                                <SelectItem value="social">Social Media</SelectItem>
                              </SelectContent>
                            </Select>
                            <FormDescription>Format for your script</FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="duration"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Duration (Optional)</FormLabel>
                            <FormControl>
                              <Input
                                placeholder="e.g., 5 minutes"
                                data-testid="input-script-duration"
                                {...field}
                              />
                            </FormControl>
                            <FormDescription>Target length</FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="tone"
                        render={({ field }) => (
                          <FormItem className="md:col-span-2">
                            <FormLabel>Tone</FormLabel>
                            <Select onValueChange={field.onChange} defaultValue={field.value}>
                              <FormControl>
                                <SelectTrigger data-testid="select-script-tone">
                                  <SelectValue placeholder="Select tone" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="professional">Professional</SelectItem>
                                <SelectItem value="casual">Casual</SelectItem>
                                <SelectItem value="educational">Educational</SelectItem>
                                <SelectItem value="entertaining">Entertaining</SelectItem>
                              </SelectContent>
                            </Select>
                            <FormDescription>Style and tone of the script</FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </div>

                    <Button
                      type="submit"
                      className="w-full"
                      disabled={generateScriptMutation.isPending}
                      data-testid="button-generate-script"
                    >
                      {generateScriptMutation.isPending ? (
                        <>
                          <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <FileText className="mr-2 h-5 w-5" />
                          Generate Script
                        </>
                      )}
                    </Button>
                  </form>
                </Form>
              </CardContent>
            </Card>

            {selectedScript && (
              <Card data-testid="card-generated-script">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>{selectedScript.title}</CardTitle>
                      <p className="text-sm text-muted-foreground mt-1">
                        {selectedScript.type}
                      </p>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleCopy(selectedScript)}
                        data-testid="button-copy-script"
                      >
                        <Copy className="h-4 w-4 mr-2" />
                        Copy
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => form.handleSubmit(onSubmit)()}
                        disabled={generateScriptMutation.isPending}
                        data-testid="button-regenerate-script"
                      >
                        <RotateCcw className="h-4 w-4 mr-2" />
                        Regenerate
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div
                    className="prose prose-sm max-w-none dark:prose-invert font-mono text-sm p-4 rounded-lg bg-muted/30 border border-card-border whitespace-pre-wrap"
                    data-testid="text-script-content"
                  >
                    {selectedScript.content}
                  </div>
                </CardContent>
              </Card>
            )}

            {!selectedScript && !generateScriptMutation.isPending && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="rounded-full bg-primary/10 p-4 mb-4">
                  <FileText className="h-8 w-8 text-primary" />
                </div>
                <h2 className="text-2xl font-semibold mb-2">Create your script</h2>
                <p className="text-muted-foreground max-w-md">
                  Fill in the details above and generate a professional script tailored to your needs.
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
                <h2 className="text-2xl font-semibold mb-2">Failed to load scripts</h2>
                <p className="text-muted-foreground max-w-md mb-4">
                  {error instanceof Error ? error.message : "An error occurred while loading your script history."}
                </p>
                <Button
                  variant="outline"
                  onClick={() => queryClient.invalidateQueries({ queryKey: ["/api/scripts"] })}
                  data-testid="button-retry-scripts"
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
            ) : scripts.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <div className="rounded-full bg-primary/10 p-4 mb-4">
                  <History className="h-8 w-8 text-primary" />
                </div>
                <h2 className="text-2xl font-semibold mb-2">No scripts yet</h2>
                <p className="text-muted-foreground max-w-md">
                  Generate your first script to see it here.
                </p>
              </div>
            ) : (
              scripts.map((script) => (
                <Card key={script.id} data-testid={`script-${script.id}`}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>{script.title}</CardTitle>
                        <p className="text-xs text-muted-foreground mt-1">
                          {script.type} •{" "}
                          {new Date(script.timestamp).toLocaleDateString()} at{" "}
                          {new Date(script.timestamp).toLocaleTimeString([], {
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleCopy(script)}
                          data-testid={`button-copy-${script.id}`}
                        >
                          <Copy className="h-4 w-4 mr-2" />
                          Copy
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => exportScriptAsPDF(script)}
                          data-testid={`button-export-pdf-${script.id}`}
                        >
                          <Download className="h-4 w-4 mr-2" />
                          PDF
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => exportScriptAsTXT(script)}
                          data-testid={`button-export-txt-${script.id}`}
                        >
                          <FileText className="h-4 w-4 mr-2" />
                          TXT
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="prose prose-sm max-w-none dark:prose-invert font-mono text-sm p-4 rounded-lg bg-muted/30 border border-card-border whitespace-pre-wrap line-clamp-6">
                      {script.content}
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
