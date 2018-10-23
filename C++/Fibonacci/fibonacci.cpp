#include <iostream>
#include <string>

using namespace std;

int main()
{
	int n = 10;
	int num[n];
	
	if (n > 1)
	{
		num[0] = 1;
	}
	if (n > 2)
	{
		num[1] = 1;
	}
	for (int i = 2; i < n; i++)
	{
		num[i] = num[i-1] + num[i-2];
		
	}	

	for (int k = 0; k < n; k++)
	{
		cout << to_string(k+1) + " : " + to_string(num[k]) + "\n";
	}
	
	return 0;

}
